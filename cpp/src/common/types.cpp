// SPDX-License-Identifier: AGPL-3.0-or-later
#include "common/types.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>

namespace bev {
Tensor::Tensor(std::array<int, 4> dimensions) : shape(dimensions) {
    std::size_t elements = 1;
    for (const auto size : shape) {
        if (size <= 0)
            throw std::invalid_argument("Tensor dimensions must be positive");
        elements *= static_cast<std::size_t>(size);
    }
    data.resize(elements);
}

std::string Sha256(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read " + path.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(),
                                                                   EVP_MD_CTX_free);
    if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::array<char, 65536> buffer{};
    while (input.read(buffer.data(), buffer.size()) || input.gcount() > 0) {
        if (EVP_DigestUpdate(digest.get(), buffer.data(), input.gcount()) != 1)
            throw std::runtime_error("Cannot update SHA256");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int size = 0;
    if (!input.eof() || EVP_DigestFinal_ex(digest.get(), hash.data(), &size) != 1)
        throw std::runtime_error("Cannot finish SHA256");
    std::ostringstream text;
    for (unsigned int i = 0; i < size; ++i)
        text << std::hex << std::setfill('0') << std::setw(2) << unsigned(hash[i]);
    return text.str();
}

void SaveTensor(const std::filesystem::path& directory, const std::string& name,
                const Tensor& tensor) {
    std::filesystem::create_directories(directory);
    std::ofstream binary(directory / (name + ".bin"), std::ios::binary);
    binary.exceptions(std::ios::failbit | std::ios::badbit);
    binary.write(reinterpret_cast<const char*>(tensor.data.data()),
                 tensor.data.size() * sizeof(float));
    const auto path = directory / "tensors.json";
    nlohmann::json metadata = nlohmann::json::object();
    if (std::filesystem::exists(path)) {
        std::ifstream previous(path);
        previous >> metadata;
    }
    metadata[name] = {{"shape", tensor.shape}, {"dtype", "float32"}, {"file", name + ".bin"}};
    std::ofstream description(path);
    description.exceptions(std::ios::failbit | std::ios::badbit);
    description << metadata.dump(2) << '\n';
}
} // namespace bev
