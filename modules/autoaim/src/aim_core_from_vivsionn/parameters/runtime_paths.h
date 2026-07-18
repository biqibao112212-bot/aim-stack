#ifndef RM_RUNTIME_PATHS_H
#define RM_RUNTIME_PATHS_H

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace rm::runtime_paths
{

namespace fs = std::filesystem;

inline fs::path normalizePath(const fs::path& path)
{
    if (path.empty()) return {};

    std::error_code ec;
    const fs::path weak = fs::weakly_canonical(path, ec);
    if (!ec) return weak;

    ec.clear();
    const fs::path absolute = fs::absolute(path, ec);
    if (!ec) return absolute.lexically_normal();

    return path.lexically_normal();
}

inline bool pathExists(const fs::path& path)
{
    std::error_code ec;
    return !path.empty() && fs::exists(path, ec);
}

inline void appendUniquePath(std::vector<fs::path>& paths, const fs::path& path)
{
    if (path.empty()) return;

    const fs::path normalized = normalizePath(path);
    if (std::find(paths.begin(), paths.end(), normalized) == paths.end()) {
        paths.push_back(normalized);
    }
}

inline fs::path executablePath()
{
#if defined(__linux__)
    std::error_code ec;
    const fs::path target = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return normalizePath(target);
#endif
    return {};
}

inline bool looksLikeRepoRoot(const fs::path& candidate)
{
    return pathExists(candidate / "CMakeLists.txt") &&
           pathExists(candidate / "src") &&
           (pathExists(candidate / "src" / "param.yaml") ||
            pathExists(candidate / "src" / "aim_core_from_vivsionn" / "param.yaml"));
}

inline fs::path legacySrcPathToAimCorePath(const fs::path& relative_path)
{
    auto it = relative_path.begin();
    const auto end = relative_path.end();
    if (it == end || it->string() != "src") return {};

    fs::path mapped = "src";
    mapped /= "aim_core_from_vivsionn";
    ++it;
    for (; it != end; ++it) {
        mapped /= *it;
    }
    return mapped;
}

inline fs::path repoRoot()
{
    static const fs::path cached_root = []() {
        std::vector<fs::path> search_seeds;

        std::error_code ec;
        const fs::path cwd = fs::current_path(ec);
        if (!ec) appendUniquePath(search_seeds, cwd);

        const fs::path exe_path = executablePath();
        if (!exe_path.empty()) {
            appendUniquePath(search_seeds, exe_path.parent_path());
            appendUniquePath(search_seeds, exe_path.parent_path().parent_path());
        }

        for (const auto& seed : search_seeds) {
            fs::path current = seed;
            while (!current.empty()) {
                if (looksLikeRepoRoot(current)) return current;

                const fs::path parent = current.parent_path();
                if (parent == current) break;
                current = parent;
            }
        }

        if (!search_seeds.empty()) return search_seeds.front();
        return normalizePath(".");
    }();

    return cached_root;
}

inline fs::path repoPath(const fs::path& relative_or_absolute_path)
{
    if (relative_or_absolute_path.empty()) return repoRoot();
    if (relative_or_absolute_path.is_absolute()) return normalizePath(relative_or_absolute_path);
    const fs::path mapped_legacy_src = legacySrcPathToAimCorePath(relative_or_absolute_path);
    if (!mapped_legacy_src.empty()) return normalizePath(repoRoot() / mapped_legacy_src);
    return normalizePath(repoRoot() / relative_or_absolute_path);
}

inline std::vector<fs::path> candidatePaths(const fs::path& relative_or_absolute_path)
{
    if (relative_or_absolute_path.empty()) return {};
    if (relative_or_absolute_path.is_absolute()) {
        return {normalizePath(relative_or_absolute_path)};
    }

    std::vector<fs::path> candidates;

    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec) appendUniquePath(candidates, cwd / relative_or_absolute_path);

    const fs::path exe_path = executablePath();
    if (!exe_path.empty()) {
        const fs::path exe_dir = exe_path.parent_path();
        appendUniquePath(candidates, exe_dir / relative_or_absolute_path);
        appendUniquePath(candidates, exe_dir.parent_path() / relative_or_absolute_path);
    }

    appendUniquePath(candidates, repoRoot() / relative_or_absolute_path);
    return candidates;
}

inline fs::path resolveExistingPath(const fs::path& relative_or_absolute_path)
{
    for (const auto& candidate : candidatePaths(relative_or_absolute_path)) {
        if (pathExists(candidate)) return candidate;
    }
    return repoPath(relative_or_absolute_path);
}

}  // namespace rm::runtime_paths

#endif  // RM_RUNTIME_PATHS_H
