/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/SegmentedFileCodesStorage.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace faiss {

namespace {

std::string codes_dir(const std::string& base) {
    return base + ".codes";
}
std::string graph_dir(const std::string& base) {
    return base + ".graph";
}
std::string meta_path(const std::string& base) {
    return base + ".meta.json";
}
std::string lock_path(const std::string& base) {
    return base + ".lock";
}

std::string seg_path(const std::string& base, size_t i) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/seg-%08zu.bin", i);
    return codes_dir(base) + buf;
}

std::string graph_file(const std::string& base, uint64_t gen) {
    char buf[64];
    std::snprintf(
            buf,
            sizeof(buf),
            "/graph-%08llu.bin",
            (unsigned long long)gen);
    return graph_dir(base) + buf;
}

void mkdir_p(const std::string& path) {
    if (path.empty() || path == "." || path == "/") {
        return;
    }
    struct stat st;
    if (::stat(path.c_str(), &st) == 0) {
        FAISS_THROW_IF_NOT_FMT(
                S_ISDIR(st.st_mode), "not a dir: %s", path.c_str());
        return;
    }
    auto i = path.find_last_of('/');
    if (i != std::string::npos && i > 0) {
        mkdir_p(path.substr(0, i));
    }
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        FAISS_THROW_FMT("mkdir(%s): %s", path.c_str(), std::strerror(errno));
    }
}

void read_segment_into(const std::string& path, uint8_t* dst, size_t n) {
    FILE* fp = ::fopen(path.c_str(), "rb");
    FAISS_THROW_IF_NOT_FMT(
            fp, "open(%s): %s", path.c_str(), std::strerror(errno));
    if (n > 0) {
        size_t r = ::fread(dst, 1, n, fp);
        FAISS_THROW_IF_NOT_FMT(
                r == n,
                "short read on %s: %zu/%zu",
                path.c_str(),
                r,
                n);
    }
    ::fclose(fp);
}

} // anonymous namespace

SegmentedFileCodesStorage::SegmentedFileCodesStorage(
        std::string basepath, size_t code_size, Options opts)
        : SegmentedFileCodesStorage(
                  std::move(basepath),
                  code_size,
                  std::unique_ptr<MetadataStore>(),
                  opts) {}

SegmentedFileCodesStorage::SegmentedFileCodesStorage(
        std::string basepath,
        size_t code_size,
        std::unique_ptr<MetadataStore> meta_store,
        Options opts)
        : basepath_(std::move(basepath)),
          code_size_(code_size),
          opts_(opts),
          meta_store_(std::move(meta_store)) {
    FAISS_THROW_IF_NOT(code_size_ > 0);
    if (!meta_store_) {
        auto slash = basepath_.find_last_of('/');
        if (slash != std::string::npos) {
            mkdir_p(basepath_.substr(0, slash));
        }
        meta_store_ = std::make_unique<JsonFileMetadataStore>(
                meta_path(basepath_), opts_.fsync_files);
    }
    hydrate_();
}

SegmentedFileCodesStorage::~SegmentedFileCodesStorage() {
    release_lock_();
}

void SegmentedFileCodesStorage::hydrate_() {
    AppendableMetadata m;
    if (!meta_store_->load(m)) {
        last_committed_ = {};
        last_committed_.code_size = code_size_;
        buffer_.clear();
        return;
    }
    FAISS_THROW_IF_NOT_FMT(
            m.code_size == code_size_,
            "meta code_size %llu != ctor code_size %zu",
            (unsigned long long)m.code_size,
            code_size_);
    last_committed_ = m;
    uint64_t total = 0;
    for (auto s : m.segment_sizes) {
        total += s;
    }
    buffer_.resize(total);
    uint8_t* dst = buffer_.data();
    for (size_t i = 0; i < m.segment_sizes.size(); ++i) {
        read_segment_into(seg_path(basepath_, i), dst, m.segment_sizes[i]);
        dst += m.segment_sizes[i];
    }
}

uint64_t SegmentedFileCodesStorage::committed_bytes() const {
    uint64_t s = 0;
    for (auto x : last_committed_.segment_sizes) {
        s += x;
    }
    return s;
}

void SegmentedFileCodesStorage::append(size_t n, const uint8_t* src) {
    if (n == 0) {
        return;
    }
    size_t old = buffer_.size();
    buffer_.resize(old + n * code_size_);
    std::memcpy(buffer_.data() + old, src, n * code_size_);
}

void SegmentedFileCodesStorage::reset() {
    buffer_.clear();
}

void SegmentedFileCodesStorage::permute(const idx_t* perm) {
    const size_t n = num_codes();
    std::vector<uint8_t> next(buffer_.size());
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(
                next.data() + i * code_size_,
                buffer_.data() + perm[i] * code_size_,
                code_size_);
    }
    buffer_.swap(next);
}

std::optional<CodesView> SegmentedFileCodesStorage::try_view() const {
    CodesView v;
    v.data = buffer_.data();
    v.nbytes = buffer_.size();
    return v;
}

void SegmentedFileCodesStorage::acquire_lock_() {
    if (lock_fd_ >= 0) {
        return;
    }
    auto slash = basepath_.find_last_of('/');
    if (slash != std::string::npos) {
        mkdir_p(basepath_.substr(0, slash));
    }
    int fd = ::open(lock_path(basepath_).c_str(), O_CREAT | O_RDWR, 0644);
    FAISS_THROW_IF_NOT_FMT(
            fd >= 0,
            "open(%s): %s",
            lock_path(basepath_).c_str(),
            std::strerror(errno));
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        FAISS_THROW_FMT(
                "another writer holds %s", lock_path(basepath_).c_str());
    }
    lock_fd_ = fd;
}

void SegmentedFileCodesStorage::release_lock_() {
    if (lock_fd_ < 0) {
        return;
    }
    ::flock(lock_fd_, LOCK_UN);
    ::close(lock_fd_);
    lock_fd_ = -1;
}

void SegmentedFileCodesStorage::maybe_kill_(const char* phase) const {
    const char* env = ::getenv("FAISS_APPENDABLE_KILL_AFTER");
    if (env && std::strcmp(env, phase) == 0) {
        ::_exit(137);
    }
}

namespace {

void write_segment_file(
        const std::string& path,
        const uint8_t* data,
        size_t n,
        bool do_fsync) {
    std::string tmp = path + ".tmp";
    FILE* fp = ::fopen(tmp.c_str(), "wb");
    FAISS_THROW_IF_NOT_FMT(
            fp, "open(%s): %s", tmp.c_str(), std::strerror(errno));
    if (n > 0) {
        size_t w = ::fwrite(data, 1, n, fp);
        FAISS_THROW_IF_NOT_FMT(
                w == n,
                "short write on %s: %zu/%zu",
                tmp.c_str(),
                w,
                n);
    }
    if (do_fsync) {
        std::fflush(fp);
        ::fsync(::fileno(fp));
    }
    std::fclose(fp);
    int rc = ::rename(tmp.c_str(), path.c_str());
    FAISS_THROW_IF_NOT_FMT(
            rc == 0,
            "rename(%s -> %s): %s",
            tmp.c_str(),
            path.c_str(),
            std::strerror(errno));
}

uint32_t inner_fourcc_of(const Index* idx) {
    VectorIOWriter w;
    write_index(idx, &w, IO_FLAG_SKIP_CODE_BYTES);
    FAISS_THROW_IF_NOT(w.data.size() >= sizeof(uint32_t));
    uint32_t fcc;
    std::memcpy(&fcc, w.data.data(), sizeof(uint32_t));
    return fcc;
}

uint64_t file_size(const std::string& p) {
    struct stat st;
    if (::stat(p.c_str(), &st) != 0) {
        return 0;
    }
    return (uint64_t)st.st_size;
}

void unlink_if_exists(const std::string& p) {
    ::unlink(p.c_str());
}

} // anonymous namespace

void SegmentedFileCodesStorage::flush(const Index* idx) {
    FAISS_THROW_IF_NOT(idx != nullptr);
    acquire_lock_();
    mkdir_p(codes_dir(basepath_));
    mkdir_p(graph_dir(basepath_));

    const size_t current_bytes = buffer_.size();
    const size_t committed = (size_t)committed_bytes();
    const uint32_t cur_fcc = inner_fourcc_of(idx);

    bool full_rewrite = last_committed_.segment_sizes.empty() ||
            last_committed_.inner_fourcc != cur_fcc ||
            last_committed_.code_size != code_size_ ||
            last_committed_.segment_bytes_target !=
                    opts_.segment_bytes_target ||
            committed > current_bytes;

    AppendableMetadata next;
    next.format_version = 1;
    next.inner_fourcc = cur_fcc;
    next.ntotal = current_bytes / code_size_;
    next.code_size = code_size_;
    next.segment_bytes_target = opts_.segment_bytes_target;
    next.graph_generation = last_committed_.graph_generation + 1;

    const size_t per_seg =
            (opts_.segment_bytes_target / code_size_) * code_size_;
    FAISS_THROW_IF_NOT_MSG(
            per_seg > 0,
            "segment_bytes_target smaller than code_size: no codes fit");

    size_t start_seg_id;
    size_t start_offset;
    if (full_rewrite) {
        for (size_t i = 0; i < last_committed_.segment_sizes.size(); ++i) {
            unlink_if_exists(seg_path(basepath_, i));
        }
        start_seg_id = 0;
        start_offset = 0;
        next.segment_sizes.clear();
    } else {
        start_seg_id = last_committed_.segment_sizes.size();
        start_offset = committed;
        next.segment_sizes = last_committed_.segment_sizes;
    }

    size_t off = start_offset;
    size_t seg_id = start_seg_id;
    while (off < current_bytes) {
        size_t chunk = std::min(per_seg, current_bytes - off);
        write_segment_file(
                seg_path(basepath_, seg_id),
                buffer_.data() + off,
                chunk,
                opts_.fsync_files);
        maybe_kill_("seg_rename");
        next.segment_sizes.push_back(chunk);
        off += chunk;
        ++seg_id;
    }
    maybe_kill_("seg_write_partial");

    std::string g_path = graph_file(basepath_, next.graph_generation);
    std::string g_tmp = g_path + ".tmp";
    {
        FileIOWriter w(g_tmp.c_str());
        write_index(idx, &w, IO_FLAG_SKIP_CODE_BYTES);
        if (opts_.fsync_files) {
            std::fflush(w.f);
            ::fsync(::fileno(w.f));
        }
    }
    maybe_kill_("graph_write");
    int rc = ::rename(g_tmp.c_str(), g_path.c_str());
    FAISS_THROW_IF_NOT_FMT(
            rc == 0,
            "rename(%s -> %s): %s",
            g_tmp.c_str(),
            g_path.c_str(),
            std::strerror(errno));
    maybe_kill_("graph_rename");
    next.graph_file_size = file_size(g_path);

    meta_store_->commit(next);
    maybe_kill_("meta_commit");

    if (last_committed_.graph_generation > 0) {
        unlink_if_exists(
                graph_file(basepath_, last_committed_.graph_generation));
    }
    DIR* d = ::opendir(codes_dir(basepath_).c_str());
    if (d) {
        struct dirent* ent;
        while ((ent = ::readdir(d)) != nullptr) {
            if (ent->d_name[0] == '.') {
                continue;
            }
            std::string name = ent->d_name;
            if (name.size() != 16 || name.compare(0, 4, "seg-") != 0 ||
                name.compare(12, 4, ".bin") != 0) {
                continue;
            }
            size_t id = std::strtoull(name.c_str() + 4, nullptr, 10);
            if (id >= next.segment_sizes.size()) {
                ::unlink((codes_dir(basepath_) + "/" + name).c_str());
            }
        }
        ::closedir(d);
    }
    maybe_kill_("gc");

    last_committed_ = next;
    release_lock_();
}

} // namespace faiss
