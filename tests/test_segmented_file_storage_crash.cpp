/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/CodesStorage.h>
#include <faiss/impl/SegmentedFileCodesStorage.h>
#include <faiss/index_io.h>

namespace {

constexpr const char* kBase = "/tmp/appendable_crash_test";

std::string fresh(const char* name) {
    ::mkdir(kBase, 0755);
    std::string p = std::string(kBase) + "/" + name;
    std::string cmd = "rm -rf " + p + " " + p + ".*";
    (void)std::system(cmd.c_str());
    return p;
}

void child_first_flush(
        const std::string& path,
        const char* kill_phase) {
    ::setenv("FAISS_APPENDABLE_KILL_AFTER", kill_phase, 1);
    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 256;
    opts.fsync_files = true;
    auto storage = std::make_shared<faiss::SegmentedFileCodesStorage>(
            path, 16, opts);
    auto* inner = new faiss::IndexFlatL2(4, storage);
    faiss::IndexHNSWFlat hnsw(4, 16);
    delete hnsw.storage;
    hnsw.storage = inner;
    hnsw.own_fields = true;
    std::vector<float> data(20 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = float(i);
    }
    hnsw.add(20, data.data());
    storage->flush(&hnsw);
}

void run_crash_test(const char* phase, bool expect_committed) {
    auto path = fresh(("crash_" + std::string(phase)).c_str());
    // Use gtest death-test "threadsafe" mode: fork+exec yields a fresh
    // process with a clean OpenMP runtime (iomp5 corrupts a forked-only
    // child that touched threads in the parent).
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
            child_first_flush(path, phase),
            ::testing::ExitedWithCode(137),
            "");

    auto storage = std::make_shared<faiss::SegmentedFileCodesStorage>(
            path, 16);
    if (expect_committed) {
        EXPECT_EQ(storage->num_codes(), 20u);
    } else {
        EXPECT_EQ(storage->num_codes(), 0u);
    }
}

} // namespace

TEST(CrashSafety, SegWritePartial) {
    run_crash_test("seg_write_partial", false);
}
TEST(CrashSafety, SegRename) {
    run_crash_test("seg_rename", false);
}
TEST(CrashSafety, GraphWrite) {
    run_crash_test("graph_write", false);
}
TEST(CrashSafety, GraphRename) {
    run_crash_test("graph_rename", false);
}
TEST(CrashSafety, MetaCommit) {
    run_crash_test("meta_commit", true);
}
TEST(CrashSafety, GC) {
    run_crash_test("gc", true);
}
