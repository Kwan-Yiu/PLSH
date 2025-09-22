#include "plsh.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>

PLSHIndex::PLSHIndex(size_t dimensions, int k, int m, unsigned int num_threads)
    : D_(dimensions),
      k_(k),
      m_(m),
      L_(m > 1 ? m * (m - 1) / 2 : 0),
      num_threads_(num_threads) {
    if (D_ == 0) {
        throw std::invalid_argument(
            "Vector dimensions must be greater than 0.");
    }
    if (k_ <= 0 || k_ % 2 != 0) {
        throw std::invalid_argument("k must be a positive even integer.");
    }
    if (m_ < 2) {
        throw std::invalid_argument("m must be 2 or greater to form pairs.");
    }
    if (num_threads_ == 0) {
        throw std::invalid_argument("Number of threads must be at least 1.");
    }

    const size_t total_hyperplanes = m_ * (k_ / 2);
    random_hyperplanes_.resize(total_hyperplanes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_hyperplanes; ++i) {
        random_hyperplanes_[i].resize(D_);
        float norm_sq = 0.0f;

        for (size_t j = 0; j < D_; ++j) {
            float val = dist(gen);
            random_hyperplanes_[i][j] = val;
            norm_sq += val * val;
        }

        float norm = std::sqrt(norm_sq);
        if (norm > 0) {
            for (size_t j = 0; j < D_; ++j) {
                random_hyperplanes_[i][j] /= norm;
            }
        }
    }

    static_tables_offsets_.resize(L_);
    static_tables_data_.resize(L_);
    delta_tables_.resize(L_);
}

void PLSHIndex::build(const std::vector<SparseVector>& data_points) {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    data_storage_.clear();
    static_tables_offsets_.clear();
    static_tables_data_.clear();
    delta_tables_.clear();

    static_tables_offsets_.resize(L_);
    static_tables_data_.resize(L_);
    delta_tables_.resize(L_);

    const size_t n_points = data_points.size();
    if (n_points == 0) {
        return;
    }
    data_storage_ = data_points;

    std::vector<std::vector<uint16_t>> base_hashes =
        _compute_base_hashes(data_storage_);
    _build_static_tables_parallel(base_hashes);
}

std::vector<std::vector<uint16_t>> PLSHIndex::_compute_base_hashes(
    const std::vector<SparseVector>& points) const {
    const size_t n_points = points.size();
    const int k_half = k_ / 2;

    std::vector<std::vector<uint16_t>> hashes(n_points,
                                              std::vector<uint16_t>(m_, 0));
    std::vector<std::thread> threads;
    size_t chunk_size = (n_points + num_threads_ - 1) / num_threads_;

    for (unsigned int t = 0; t < num_threads_; ++t) {
        threads.emplace_back([=, &points, &hashes] {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n_points);

            for (size_t i = start; i < end; ++i) {
                for (int j = 0; j < m_; ++j) {
                    uint16_t current_hash = 0;
                    for (int bit = 0; bit < k_half; ++bit) {
                        size_t hyperplane_idx = j * k_half + bit;
                        float dot_product = 0.0f;

                        for (size_t k = 0; k < points[i].indices.size(); ++k) {
                            uint32_t feature_idx = points[i].indices[k];
                            dot_product += points[i].values[k] *
                                           random_hyperplanes_[hyperplane_idx]
                                                              [feature_idx];
                        }

                        if (dot_product >= 0) {
                            current_hash |= (1 << bit);
                        }
                    }
                    hashes[i][j] = current_hash;
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    return hashes;
}

void PLSHIndex::_build_static_tables_parallel(
    const std::vector<std::vector<uint16_t>>& base_hashes) {
    const size_t n_points = data_storage_.size();
    if (n_points == 0) return;

    const int num_partitions_l1 = 1 << (k_ / 2);
    std::vector<std::vector<uint32_t>> level1_partitions(
        m_, std::vector<uint32_t>(n_points));

    std::vector<std::thread> threads_l1;
    for (int i = 0; i < m_; ++i) {
        threads_l1.emplace_back([this, i, n_points, num_partitions_l1,
                                 &level1_partitions, &base_hashes] {
            std::vector<uint32_t>& output_indices = level1_partitions[i];

            std::vector<std::vector<uint32_t>> local_histograms(
                num_threads_, std::vector<uint32_t>(num_partitions_l1, 0));
            size_t chunk_size = (n_points + num_threads_ - 1) / num_threads_;

            std::vector<std::thread> hist_threads;
            for (unsigned int t = 0; t < num_threads_; ++t) {
                hist_threads.emplace_back([this, t, i, chunk_size, n_points,
                                           &local_histograms, &base_hashes] {
                    size_t start = t * chunk_size;
                    size_t end = std::min(start + chunk_size, n_points);
                    for (size_t p = start; p < end; ++p) {
                        local_histograms[t][base_hashes[p][i]]++;
                    }
                });
            }
            for (auto& th : hist_threads) th.join();

            std::vector<uint32_t> global_offsets(num_partitions_l1, 0);
            for (unsigned int t = 0; t < num_threads_; ++t) {
                for (int part = 0; part < num_partitions_l1; ++part) {
                    uint32_t count = local_histograms[t][part];
                    local_histograms[t][part] = global_offsets[part];
                    global_offsets[part] += count;
                }
            }

            std::vector<std::thread> scatter_threads;
            for (unsigned int t = 0; t < num_threads_; ++t) {
                scatter_threads.emplace_back([this, t, i, chunk_size, n_points,
                                              &local_histograms, &base_hashes,
                                              &output_indices] {
                    size_t start = t * chunk_size;
                    size_t end = std::min(start + chunk_size, n_points);
                    for (size_t p = start; p < end; ++p) {
                        uint16_t hash_val = base_hashes[p][i];
                        output_indices[local_histograms[t][hash_val]++] = p;
                    }
                });
            }
            for (auto& th : scatter_threads) th.join();
        });
    }
    for (auto& th : threads_l1) th.join();

    const int num_partitions_l2 = 1 << k_;
    int table_idx = 0;
    for (int i = 0; i < m_; ++i) {
        for (int j = i + 1; j < m_; ++j) {
            static_tables_data_[table_idx].resize(n_points);
            static_tables_offsets_[table_idx].resize(num_partitions_l2 + 1, 0);
            const std::vector<uint32_t>& input_indices = level1_partitions[i];
            std::vector<uint16_t> reordered_l2_hashes(n_points);
            for (size_t p = 0; p < n_points; ++p) {
                reordered_l2_hashes[p] = base_hashes[input_indices[p]][j];
            }

            auto& offsets = static_tables_offsets_[table_idx];
            for (size_t p = 0; p < n_points; ++p) {
                uint16_t l1_hash = base_hashes[input_indices[p]][i];
                uint16_t l2_hash = reordered_l2_hashes[p];
                uint32_t combined_hash =
                    (static_cast<uint32_t>(l1_hash) << (k_ / 2)) | l2_hash;
                offsets[combined_hash + 1]++;
            }

            for (int part = 0; part < num_partitions_l2; ++part) {
                offsets[part + 1] += offsets[part];
            }

            auto& data_output = static_tables_data_[table_idx];
            std::vector<uint32_t> current_offsets = offsets;
            for (size_t p = 0; p < n_points; ++p) {
                uint16_t l1_hash = base_hashes[input_indices[p]][i];
                uint16_t l2_hash = reordered_l2_hashes[p];
                uint32_t combined_hash =
                    (static_cast<uint32_t>(l1_hash) << (k_ / 2)) | l2_hash;
                data_output[current_offsets[combined_hash]++] =
                    input_indices[p];
            }

            table_idx++;
        }
    }
}

void PLSHIndex::insert(const SparseVector& data_point) {
    std::lock_guard<std::mutex> lock(delta_insert_mutex_);
    const uint32_t new_id = data_storage_.size();
    data_storage_.push_back(data_point);
    const int k_half = k_ / 2;
    std::vector<uint16_t> base_hashes(m_);

    for (int i = 0; i < m_; ++i) {
        uint16_t current_hash = 0;
        for (int bit = 0; bit < k_half; ++bit) {
            size_t hyperplane_idx = i * k_half + bit;
            float dot_product = 0.0f;
            for (size_t k = 0; k < data_point.indices.size(); ++k) {
                uint32_t feature_idx = data_point.indices[k];
                if (feature_idx < D_) {
                    dot_product +=
                        data_point.values[k] *
                        random_hyperplanes_[hyperplane_idx][feature_idx];
                }
            }

            if (dot_product >= 0) {
                current_hash |= (1 << bit);
            }
        }
        base_hashes[i] = current_hash;
    }

    int table_idx = 0;
    for (int i = 0; i < m_; ++i) {
        for (int j = i + 1; j < m_; ++j) {
            uint32_t combined_hash =
                (static_cast<uint32_t>(base_hashes[i]) << k_half) |
                base_hashes[j];
            if (delta_tables_[table_idx].empty()) {
                delta_tables_[table_idx].resize(1 << k_);
            }
            delta_tables_[table_idx][combined_hash].push_back(new_id);
            table_idx++;
        }
    }
}

std::vector<Result> PLSHIndex::query_radius(const SparseVector& query_point,
                                            float radius) const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    std::vector<uint32_t> candidates = _get_candidates(query_point);

    if (candidates.empty()) {
        return {};
    }

    std::vector<uint32_t> unique_candidates;
    unique_candidates.reserve(candidates.size());
    std::vector<bool> seen(data_storage_.size(), false);

    for (const uint32_t id : candidates) {
        if (!seen[id]) {
            unique_candidates.push_back(id);
            seen[id] = true;
        }
    }
    return _filter_candidates(query_point, unique_candidates, radius);
}

std::vector<Result> PLSHIndex::query_topk(const SparseVector& query_point,
                                          size_t topk) const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    std::vector<uint32_t> candidates = _get_candidates(query_point);

    if (candidates.empty()) {
        return {};
    }

    // 去重
    std::vector<uint32_t> unique_candidates;
    unique_candidates.reserve(candidates.size());
    std::vector<bool> seen(data_storage_.size(), false);
    for (const uint32_t id : candidates) {
        if (!seen[id]) {
            unique_candidates.push_back(id);
            seen[id] = true;
        }
    }

    // 归一化 query
    SparseVector normalized_query = query_point;
    float norm_sq = 0.0f;
    for (float val : normalized_query.values) norm_sq += val * val;
    float norm = std::sqrt(norm_sq);
    if (norm > 0) {
        for (float& val : normalized_query.values) val /= norm;
    }

    // 计算所有候选的角距离
    std::vector<Result> results;
    results.reserve(unique_candidates.size());
    for (const uint32_t id : unique_candidates) {
        const SparseVector& candidate_vec = data_storage_[id];
        float distance = l2_distance(normalized_query, candidate_vec);
        results.push_back({id, distance});
    }

    // Top-K 选择
    if (results.size() > topk) {
        std::nth_element(results.begin(), results.begin() + topk, results.end(),
                         [](const Result& a, const Result& b) {
                             return a.distance < b.distance;
                         });
        results.resize(topk);
        std::sort(results.begin(), results.end(),
                  [](const Result& a, const Result& b) {
                      return a.distance < b.distance;
                  });
    } else {
        std::sort(results.begin(), results.end(),
                  [](const Result& a, const Result& b) {
                      return a.distance < b.distance;
                  });
    }

    return results;
}

std::vector<uint32_t> PLSHIndex::_get_candidates(
    const SparseVector& query_point) const {
    std::vector<uint32_t> candidates;
    const int k_half = k_ / 2;

    std::vector<uint16_t> base_hashes(m_);
    for (int i = 0; i < m_; ++i) {
        uint16_t current_hash = 0;
        for (int bit = 0; bit < k_half; ++bit) {
            size_t hyperplane_idx = i * k_half + bit;
            float dot_product = 0.0f;
            for (size_t k = 0; k < query_point.indices.size(); ++k) {
                uint32_t feature_idx = query_point.indices[k];
                if (feature_idx < D_) {
                    dot_product +=
                        query_point.values[k] *
                        random_hyperplanes_[hyperplane_idx][feature_idx];
                }
            }
            if (dot_product >= 0) {
                current_hash |= (1 << bit);
            }
        }
        base_hashes[i] = current_hash;
    }

    int table_idx = 0;
    for (int i = 0; i < m_; ++i) {
        for (int j = i + 1; j < m_; ++j) {
            uint32_t combined_hash =
                (static_cast<uint32_t>(base_hashes[i]) << k_half) |
                base_hashes[j];
            if (table_idx < static_tables_offsets_.size() &&
                !static_tables_offsets_[table_idx].empty()) {
                uint32_t start =
                    static_tables_offsets_[table_idx][combined_hash];
                uint32_t end =
                    static_tables_offsets_[table_idx][combined_hash + 1];
                for (uint32_t p = start; p < end; ++p) {
                    candidates.push_back(static_tables_data_[table_idx][p]);
                }
            }

            if (table_idx < delta_tables_.size() &&
                !delta_tables_[table_idx].empty()) {
                const auto& bucket = delta_tables_[table_idx][combined_hash];
                candidates.insert(candidates.end(), bucket.begin(),
                                  bucket.end());
            }

            table_idx++;
        }
    }

    return candidates;
}

static float sparse_dot_product(const SparseVector& v1,
                                const SparseVector& v2) {
    float product = 0.0f;
    std::unordered_map<uint32_t, float> v2_map;
    for (size_t i = 0; i < v2.indices.size(); ++i) {
        v2_map[v2.indices[i]] = v2.values[i];
    }

    for (size_t i = 0; i < v1.indices.size(); ++i) {
        auto it = v2_map.find(v1.indices[i]);
        if (it != v2_map.end()) {
            product += v1.values[i] * it->second;
        }
    }
    return product;
}

float PLSHIndex::l2_distance(const SparseVector& v1, const SparseVector& v2) {
    // 假设两者都是稀疏存储
    std::unordered_map<uint32_t, float> v2_map;
    for (size_t i = 0; i < v2.indices.size(); ++i) {
        v2_map[v2.indices[i]] = v2.values[i];
    }

    float dist_sq = 0.0f;
    for (size_t i = 0; i < v1.indices.size(); ++i) {
        uint32_t idx = v1.indices[i];
        float diff = v1.values[i] - (v2_map.count(idx) ? v2_map[idx] : 0.0f);
        dist_sq += diff * diff;
        v2_map.erase(idx);
    }
    for (auto& kv : v2_map) {
        dist_sq += kv.second * kv.second;
    }
    return std::sqrt(dist_sq);
}

std::vector<Result> PLSHIndex::_filter_candidates(
    const SparseVector& query_point, const std::vector<uint32_t>& candidates,
    float radius) const {
    std::vector<Result> results;
    SparseVector normalized_query = query_point;
    float norm_sq = 0.0f;
    for (float val : normalized_query.values) norm_sq += val * val;
    float norm = std::sqrt(norm_sq);
    if (norm > 0) {
        for (float& val : normalized_query.values) val /= norm;
    }

    for (const uint32_t id : candidates) {
        const SparseVector& candidate_vec = data_storage_[id];
        float min_cosine = std::cos(radius);

        float dot_product = sparse_dot_product(normalized_query, candidate_vec);

        if (dot_product >= min_cosine) {
            float distance =
                std::acos(std::max(-1.0f, std::min(1.0f, dot_product)));
            results.push_back({id, distance});
        }
    }

    return results;
}

void PLSHIndex::merge_delta_to_static() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);

    if (data_storage_.empty()) {
        return;
    }

    static_tables_offsets_.clear();
    static_tables_data_.clear();
    delta_tables_.clear();

    static_tables_offsets_.resize(L_);
    static_tables_data_.resize(L_);
    delta_tables_.resize(L_);

    if (!data_storage_.empty()) {
        std::vector<std::vector<uint16_t>> base_hashes =
            _compute_base_hashes(data_storage_);
        _build_static_tables_parallel(base_hashes);
    }
}