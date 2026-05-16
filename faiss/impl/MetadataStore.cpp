/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/MetadataStore.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

bool InMemoryMetadataStore::load(AppendableMetadata& out) {
    if (!has_) {
        return false;
    }
    out = state_;
    return true;
}

void InMemoryMetadataStore::commit(const AppendableMetadata& m) {
    state_ = m;
    has_ = true;
}

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

void fsync_path(const std::string& p) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        return;
    }
    ::fsync(fd);
    ::close(fd);
}

std::string parent_dir(const std::string& p) {
    auto i = p.find_last_of('/');
    if (i == std::string::npos) {
        return std::string(".");
    }
    if (i == 0) {
        return std::string("/");
    }
    return p.substr(0, i);
}

std::string serialize(const AppendableMetadata& m) {
    std::ostringstream os;
    os << "{\n";
    os << "  \"format_version\": " << m.format_version << ",\n";
    os << "  \"inner_fourcc\": " << m.inner_fourcc << ",\n";
    os << "  \"ntotal\": " << m.ntotal << ",\n";
    os << "  \"code_size\": " << m.code_size << ",\n";
    os << "  \"segment_bytes_target\": " << m.segment_bytes_target << ",\n";
    os << "  \"segment_sizes\": [";
    for (size_t i = 0; i < m.segment_sizes.size(); ++i) {
        if (i) {
            os << ", ";
        }
        os << m.segment_sizes[i];
    }
    os << "],\n";
    os << "  \"graph_generation\": " << m.graph_generation << ",\n";
    os << "  \"graph_file_size\": " << m.graph_file_size << "\n";
    os << "}\n";
    return os.str();
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace((unsigned char)s[i])) {
        ++i;
    }
}

void expect(const std::string& s, size_t& i, const char* lit) {
    skip_ws(s, i);
    size_t n = std::strlen(lit);
    FAISS_THROW_IF_NOT_FMT(
            i + n <= s.size() && s.compare(i, n, lit) == 0,
            "JSON parse: expected '%s' at offset %zu",
            lit,
            i);
    i += n;
}

uint64_t read_uint(const std::string& s, size_t& i) {
    skip_ws(s, i);
    char* end = nullptr;
    uint64_t v = std::strtoull(s.c_str() + i, &end, 10);
    FAISS_THROW_IF_NOT_MSG(
            end != s.c_str() + i, "JSON parse: expected integer");
    i = static_cast<size_t>(end - s.c_str());
    return v;
}

void read_field(
        const std::string& s,
        size_t& i,
        const char* name,
        uint64_t& out) {
    expect(s, i, "\"");
    expect(s, i, name);
    expect(s, i, "\"");
    expect(s, i, ":");
    out = read_uint(s, i);
}

AppendableMetadata parse(const std::string& s) {
    AppendableMetadata m;
    size_t i = 0;
    expect(s, i, "{");
    uint64_t tmp;
    read_field(s, i, "format_version", tmp);
    m.format_version = (uint32_t)tmp;
    expect(s, i, ",");
    read_field(s, i, "inner_fourcc", tmp);
    m.inner_fourcc = (uint32_t)tmp;
    expect(s, i, ",");
    read_field(s, i, "ntotal", m.ntotal);
    expect(s, i, ",");
    read_field(s, i, "code_size", m.code_size);
    expect(s, i, ",");
    read_field(s, i, "segment_bytes_target", m.segment_bytes_target);
    expect(s, i, ",");
    expect(s, i, "\"");
    expect(s, i, "segment_sizes");
    expect(s, i, "\"");
    expect(s, i, ":");
    expect(s, i, "[");
    skip_ws(s, i);
    if (i < s.size() && s[i] != ']') {
        while (true) {
            uint64_t v = read_uint(s, i);
            m.segment_sizes.push_back(v);
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') {
                ++i;
                continue;
            }
            break;
        }
    }
    expect(s, i, "]");
    expect(s, i, ",");
    read_field(s, i, "graph_generation", m.graph_generation);
    expect(s, i, ",");
    read_field(s, i, "graph_file_size", m.graph_file_size);
    expect(s, i, "}");
    return m;
}

} // anonymous namespace

JsonFileMetadataStore::JsonFileMetadataStore(std::string path, bool fs)
        : path_(std::move(path)), fsync_(fs) {}

bool JsonFileMetadataStore::load(AppendableMetadata& out) {
    if (!file_exists(path_)) {
        return false;
    }
    FILE* fp = ::fopen(path_.c_str(), "rb");
    FAISS_THROW_IF_NOT_FMT(
            fp, "open(%s): %s", path_.c_str(), std::strerror(errno));
    std::fseek(fp, 0, SEEK_END);
    long n = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string buf(n, '\0');
    if (n > 0) {
        size_t r = std::fread(buf.data(), 1, n, fp);
        std::fclose(fp);
        FAISS_THROW_IF_NOT_FMT(
                (long)r == n, "short read on %s", path_.c_str());
    } else {
        std::fclose(fp);
    }
    out = parse(buf);
    return true;
}

void JsonFileMetadataStore::commit(const AppendableMetadata& m) {
    std::string tmp = path_ + ".tmp";
    {
        FILE* fp = ::fopen(tmp.c_str(), "wb");
        FAISS_THROW_IF_NOT_FMT(
                fp, "open(%s): %s", tmp.c_str(), std::strerror(errno));
        std::string s = serialize(m);
        size_t w = std::fwrite(s.data(), 1, s.size(), fp);
        FAISS_THROW_IF_NOT_FMT(
                w == s.size(), "short write on %s", tmp.c_str());
        if (fsync_) {
            std::fflush(fp);
            ::fsync(::fileno(fp));
        }
        std::fclose(fp);
    }
    int rc = ::rename(tmp.c_str(), path_.c_str());
    FAISS_THROW_IF_NOT_FMT(
            rc == 0,
            "rename(%s, %s): %s",
            tmp.c_str(),
            path_.c_str(),
            std::strerror(errno));
    if (fsync_) {
        fsync_path(parent_dir(path_));
    }
}

} // namespace faiss
