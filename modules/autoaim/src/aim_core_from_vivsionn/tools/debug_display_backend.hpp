#ifndef TOOLS_DEBUG_DISPLAY_BACKEND_HPP
#define TOOLS_DEBUG_DISPLAY_BACKEND_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace tools
{

inline bool show_debug_image(const std::string& name, const cv::Mat& image)
{
    if (image.empty()) return false;
    cv::imshow(name, image);
    return true;
}

inline int poll_debug_key(int delay_ms)
{
    return cv::waitKey(delay_ms);
}

}  // namespace tools

#endif  // TOOLS_DEBUG_DISPLAY_BACKEND_HPP
