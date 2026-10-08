#pragma once

#include <cstddef>
#include <string>
#include <vector>

bool TryGetParaFromInput_int(const char* fname, const char* key, int& value);
bool TryGetParaFromInput_real(const char* fname, const char* key, double& value);
bool TryGetParaFromInput_bool(const char* fname, const char* key, bool& value);

void GetParaFromInput_int(const char* fname, const char* key, int& value);
void GetParaFromInput_real(const char* fname, const char* key, double& value);

void Vec_fwrite_double(const char* fname, const double* data, std::size_t size);
void Vec_fread_double(const char* fname, double* data, std::size_t size);

std::vector<double> ParseDoubleList(const std::string& text);
std::string Trim(const std::string& text);