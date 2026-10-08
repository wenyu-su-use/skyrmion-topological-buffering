#include "common.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

    bool ReadRawValue(const char* fname, const char* key, std::string& raw_value) {
        std::ifstream input(fname);
        if (!input) {
            return false;
        }

        const std::string expected = key;
        std::string line;

        while (std::getline(input, line)) {
            const std::size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line.erase(comment);
            }

            const std::size_t equal = line.find('=');
            if (equal == std::string::npos) {
                continue;
            }

            const std::string current_key = Trim(line.substr(0, equal));
            if (current_key != expected) {
                continue;
            }

            raw_value = Trim(line.substr(equal + 1));
            return true;
        }

        return false;
    }

} // namespace

std::string Trim(const std::string& text) {
    const auto not_space = [](unsigned char ch) {
        return !std::isspace(ch);
    };

    const auto first = std::find_if(text.begin(), text.end(), not_space);
    if (first == text.end()) {
        return {};
    }

    const auto last = std::find_if(text.rbegin(), text.rend(), not_space).base();
    return std::string(first, last);
}

bool TryGetParaFromInput_int(const char* fname, const char* key, int& value) {
    std::string raw;
    if (!ReadRawValue(fname, key, raw)) {
        return false;
    }

    std::istringstream stream(raw);
    int parsed = 0;

    if (!(stream >> parsed)) {
        throw std::runtime_error(std::string("Invalid integer for key '") + key + "'.");
    }

    value = parsed;
    return true;
}

bool TryGetParaFromInput_real(const char* fname, const char* key, double& value) {
    std::string raw;
    if (!ReadRawValue(fname, key, raw)) {
        return false;
    }

    std::istringstream stream(raw);
    double parsed = 0.0;

    if (!(stream >> parsed)) {
        throw std::runtime_error(std::string("Invalid real number for key '") + key + "'.");
    }

    value = parsed;
    return true;
}

bool TryGetParaFromInput_bool(const char* fname, const char* key, bool& value) {
    std::string raw;
    if (!ReadRawValue(fname, key, raw)) {
        return false;
    }

    std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (raw == "1" || raw == "true" || raw == "yes" || raw == "on") {
        value = true;
        return true;
    }

    if (raw == "0" || raw == "false" || raw == "no" || raw == "off") {
        value = false;
        return true;
    }

    throw std::runtime_error(std::string("Invalid boolean for key '") + key + "'.");
}

void GetParaFromInput_int(const char* fname, const char* key, int& value) {
    if (!TryGetParaFromInput_int(fname, key, value)) {
        throw std::runtime_error(std::string("Missing integer parameter '") + key + "'.");
    }

    std::cout << "GetParaFromInput: " << key << " = " << value << '\n';
}

void GetParaFromInput_real(const char* fname, const char* key, double& value) {
    if (!TryGetParaFromInput_real(fname, key, value)) {
        throw std::runtime_error(std::string("Missing real parameter '") + key + "'.");
    }

    std::cout << "GetParaFromInput: " << key << " = " << value << '\n';
}

void Vec_fwrite_double(const char* fname, const double* data, std::size_t size) {
    std::ofstream output(fname, std::ios::binary);
    if (!output) {
        throw std::runtime_error(std::string("Cannot open output file: ") + fname);
    }

    output.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(size * sizeof(double))
    );

    if (!output) {
        throw std::runtime_error(std::string("Failed to write file: ") + fname);
    }
}

void Vec_fread_double(const char* fname, double* data, std::size_t size) {
    std::ifstream input(fname, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("Cannot open input file: ") + fname);
    }

    input.read(
        reinterpret_cast<char*>(data),
        static_cast<std::streamsize>(size * sizeof(double))
    );

    if (!input) {
        throw std::runtime_error(std::string("Failed to read file: ") + fname);
    }
}

std::vector<double> ParseDoubleList(const std::string& text) {
    std::string normalized = text;

    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::replace(normalized.begin(), normalized.end(), ';', ' ');

    std::istringstream stream(normalized);
    std::vector<double> values;

    double value = 0.0;
    while (stream >> value) {
        values.push_back(value);
    }

    return values;
}