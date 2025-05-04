#ifndef CMAKE_UTILS_HPP
#define CMAKE_UTILS_HPP

#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <string>

bool check_is_little_endian() {
    const uint32_t word = 1;
    return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
  }
  
constexpr uint32_t flip_endianness(uint32_t value) {
    return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
        ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream& stream) {
    static const bool is_little_endian = check_is_little_endian();
    uint32_t value = 0;
    AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
    return is_little_endian ? flip_endianness(value) : value;
  }

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
    const auto value = read_int32(stream);
    TORCH_CHECK(
        value == expected,
        "Expected to read number ",
        expected,
        " but found ",  
        value,
        " instead");
    return value;
  }

#endif