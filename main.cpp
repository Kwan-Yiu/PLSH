#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "plsh.hpp"

SparseVector create_normalized_random_vector(size_t dimensions, int nnz) {
    SparseVector vec;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, dimensions - 1);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
    float norm_sq = 0.0f;
    for (int i = 0; i < nnz; ++i) {
        vec.indices.push_back(distrib(gen));
        float val = val_dist(gen);
        vec.values.push_back(val);
        norm_sq += val * val;
    }
    float norm = std::sqrt(norm_sq);
    if (norm > 0) {
        for (float& val : vec.values) val /= norm;
    }
    return vec;
}

float calculate_angular_distance(const SparseVector& v1,
                                 const SparseVector& v2) {
    std::unordered_map<uint32_t, float> v2_map;
    for (size_t i = 0; i < v2.indices.size(); ++i) {
        v2_map[v2.indices[i]] = v2.values[i];
    }

    float dot_product = 0.0f;
    for (size_t i = 0; i < v1.indices.size(); ++i) {
        auto it = v2_map.find(v1.indices[i]);
        if (it != v2_map.end()) {
            dot_product += v1.values[i] * it->second;
        }
    }

    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
    return std::acos(dot_product);
}

std::vector<Result> find_ground_truth(const SparseVector& query_point,
                                      const std::vector<SparseVector>& all_data,
                                      float radius,
                                      uint32_t query_id_to_exclude) {
    std::vector<Result> ground_truth_results;
    for (uint32_t i = 0; i < all_data.size(); ++i) {
        if (i == query_id_to_exclude) continue;

        float distance = calculate_angular_distance(query_point, all_data[i]);
        if (distance <= radius) {
            ground_truth_results.push_back({i, distance});
        }
    }
    return ground_truth_results;
}

int main() {
    std::cout << "--- PLSH Index Full Demo with Streaming & Periodic Merge ---"
              << std::endl;

    const size_t dimensions = 1024;
    const int k = 8;   // k-bit hash (k/2 from each base hash)
    const int m = 40;  // m base hashes
    const unsigned int num_threads = std::thread::hardware_concurrency();

    const int initial_points = 50000;
    const int streaming_points = 200000;
    const int total_points = initial_points + streaming_points;
    const int merge_threshold = 5000000;

    std::cout << "\n[CONFIG]" << std::endl;
    std::cout << "  - Dimensions: " << dimensions << ", k: " << k
              << ", m: " << m << std::endl;
    std::cout << "  - Threads: " << num_threads << std::endl;
    std::cout << "  - Data Size: " << initial_points << " (initial) + "
              << streaming_points << " (streaming) = " << total_points
              << std::endl;
    std::cout << "  - Merge Threshold: " << merge_threshold << " inserts"
              << std::endl;

    try {
        std::cout << "\n[PHASE 1: Data Generation]" << std::endl;
        std::vector<SparseVector> all_data;
        all_data.reserve(total_points);
        for (int i = 0; i < total_points; ++i) {
            all_data.push_back(create_normalized_random_vector(dimensions, 20));
        }
        std::vector<SparseVector> initial_data(
            all_data.begin(), all_data.begin() + initial_points);
        std::vector<SparseVector> streaming_data(
            all_data.begin() + initial_points, all_data.end());
        std::cout << "  - Generated " << total_points << " sparse vectors."
                  << std::endl;

        PLSHIndex index(dimensions, k, m, num_threads);
        std::cout << "\n[PHASE 2: Initial Index Build]" << std::endl;
        auto build_start = std::chrono::high_resolution_clock::now();
        index.build(initial_data);
        auto build_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> build_duration = build_end - build_start;
        std::cout << "  - Built initial index with " << initial_points
                  << " points in " << build_duration.count() << "s."
                  << std::endl;

        std::cout << "\n[PHASE 3: Streaming Inserts & Periodic Merging]"
                  << std::endl;
        int insert_count_since_merge = 0;
        auto streaming_start = std::chrono::high_resolution_clock::now();

        for (const auto& point : streaming_data) {
            index.insert(point);
            insert_count_since_merge++;

            if (insert_count_since_merge == merge_threshold) {
                std::cout << "  -> Reached insert threshold. Triggering index "
                             "rebuild..."
                          << std::endl;
                auto rebuild_start = std::chrono::high_resolution_clock::now();
                index.merge_delta_to_static();
                auto rebuild_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> rebuild_duration =
                    rebuild_end - rebuild_start;
                std::cout << "  -> Rebuild finished in "
                          << rebuild_duration.count() << "s." << std::endl;
                insert_count_since_merge = 0;
            }
        }

        if (insert_count_since_merge > 0) {
            std::cout << "  -> Performing final merge for remaining "
                      << insert_count_since_merge << " points..." << std::endl;
            index.merge_delta_to_static();
        }

        auto streaming_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> streaming_duration =
            streaming_end - streaming_start;
        double effective_insert_qps =
            streaming_points / streaming_duration.count();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  - Total streaming inserts: " << streaming_points
                  << std::endl;
        std::cout << "  - Total streaming time (inserts + merges): "
                  << streaming_duration.count() << " s" << std::endl;
        std::cout << "  - Effective Insert QPS: " << effective_insert_qps
                  << " ops/sec" << std::endl;

        std::cout << "\n[PHASE 4: Query Performance Test (After all merges)]"
                  << std::endl;
        const int num_queries = 200;
        const float radius = 1.5f;

        auto query_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_queries; ++i) {
            volatile auto results = index.query(all_data[i], radius);
        }
        auto query_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> query_duration = query_end - query_start;
        double query_qps = num_queries / query_duration.count();
        double avg_latency_ms = (query_duration.count() * 1000) / num_queries;

        std::cout << "  - Total Queries: " << num_queries << std::endl;
        std::cout << "  - Total Time:    " << query_duration.count() << " s"
                  << std::endl;
        std::cout << "  - Query QPS:     " << query_qps << " ops/sec"
                  << std::endl;
        std::cout << "  - Avg Latency:   " << avg_latency_ms << " ms/query"
                  << std::endl;

        std::cout << "\n[PHASE 5: Recall Verification (for a single query)]"
                  << std::endl;
        const int query_idx_for_recall = 0;
        const SparseVector& query_point = all_data[query_idx_for_recall];

        std::vector<Result> lsh_results = index.query(query_point, radius);
        std::vector<Result> ground_truth = find_ground_truth(
            query_point, all_data, radius, query_idx_for_recall);

        std::unordered_set<uint32_t> ground_truth_ids;
        for (const auto& res : ground_truth) ground_truth_ids.insert(res.id);

        int true_positives = 0;
        for (const auto& lsh_res : lsh_results) {
            if (ground_truth_ids.count(lsh_res.id)) {
                true_positives++;
            }
        }

        double recall =
            ground_truth.empty()
                ? 1.0
                : static_cast<double>(true_positives) / ground_truth.size();
        std::cout << "  - Query Point ID: " << query_idx_for_recall
                  << std::endl;
        std::cout << "  - Ground Truth Neighbors (within radius " << radius
                  << "): " << ground_truth.size() << std::endl;
        std::cout << "  - LSH Found Neighbors (within radius " << radius
                  << "): " << lsh_results.size() << std::endl;
        std::cout << "  - Correctly Found (True Positives): " << true_positives
                  << std::endl;
        std::cout << "  - Recall: " << recall * 100.0 << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Demo finished successfully ---" << std::endl;
    return 0;
}