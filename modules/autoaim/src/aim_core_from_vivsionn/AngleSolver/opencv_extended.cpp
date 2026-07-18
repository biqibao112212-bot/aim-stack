#include "opencv_extended.h"

namespace cvex
{

void showHist(const cv::Mat & img)
{
    if(img.empty()) return;
    std::vector<cv::Mat> bgr_planes;
    cv::split(img, bgr_planes);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
    
    // [修复点] CV_MINMAX -> cv::NORM_MINMAX
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, CV_32F);

    for(int i=1; i<histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
                 cv::Scalar(255, 255, 255), 2, 8, 0);
    }
    cv::imshow("Hist", histImage);
}

cv::Rect scaleRect(const cv::Rect & rect, cv::Vec2f scale)
{
    cv::Point2f center = (rect.tl() + rect.br()) / 2;
    cv::Size2f size = rect.size();
    size.width *= scale[0];
    size.height *= scale[1];
    cv::Point2f tl = center - cv::Point2f(size.width/2, size.height/2);
    return cv::Rect(tl, size);
}

}