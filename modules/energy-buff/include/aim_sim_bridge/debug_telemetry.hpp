#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace aim_sim_bridge::debug
{

inline std::string envPath(const char* key)
{
    const char* value = std::getenv(key);
    if (value == nullptr || value[0] == '\0') return {};
    return value;
}

inline std::string escapeJson(std::string_view input)
{
    std::ostringstream out;
    for (char c : input) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out << "\\u00";
                constexpr char hex[] = "0123456789abcdef";
                out << hex[(static_cast<unsigned char>(c) >> 4) & 0x0f]
                    << hex[static_cast<unsigned char>(c) & 0x0f];
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

inline void writeJsonFile(const std::string& path_text, const std::string& json)
{
    if (path_text.empty()) return;

    const std::filesystem::path path(path_text);
    const std::filesystem::path parent = path.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

    std::filesystem::path temp = path;
    temp += ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) return;
        out << json << '\n';
    }

    std::filesystem::rename(temp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temp, path, ec);
    }
}

inline void appendJsonLine(const std::string& path_text, const std::string& json)
{
    if (path_text.empty()) return;

    const std::filesystem::path path(path_text);
    const std::filesystem::path parent = path.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) return;
    out << json << '\n';
}

inline void comma(std::ostream& out, bool& first)
{
    if (!first) out << ',';
    first = false;
}

inline void appendString(
    std::ostream& out, const char* key, const std::string& value, bool& first)
{
    comma(out, first);
    out << '"' << key << "\":\"" << escapeJson(value) << '"';
}

inline void appendBool(std::ostream& out, const char* key, bool value, bool& first)
{
    comma(out, first);
    out << '"' << key << "\":" << (value ? "true" : "false");
}

inline void appendInt(std::ostream& out, const char* key, long long value, bool& first)
{
    comma(out, first);
    out << '"' << key << "\":" << value;
}

inline void appendUInt(std::ostream& out, const char* key, unsigned long long value, bool& first)
{
    comma(out, first);
    out << '"' << key << "\":" << value;
}

inline void appendNumber(std::ostream& out, const char* key, double value, bool& first)
{
    comma(out, first);
    out << '"' << key << "\":";
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

inline void appendRaw(std::ostream& out, const char* key, const std::string& value, bool& first)
{
    comma(out, first);
    out << '"' << key << "\":" << value;
}

}  // namespace aim_sim_bridge::debug
