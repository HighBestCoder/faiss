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

void SegmentedFileCodesStorage::flush(const Index* /*idx*/) {
    FAISS_THROW_MSG("flush() not yet implemented");
}

} // namespace faiss
