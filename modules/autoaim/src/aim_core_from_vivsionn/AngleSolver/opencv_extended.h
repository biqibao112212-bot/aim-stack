#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

namespace cv
{

template<typename T1, typename T2>
inline Point_<T1> operator/(const cv::Point_<T1>& pt, const T2 den)
{
	return Point_<T1>(pt.x / den, pt.y / den);
}

typedef cv::Rect_<float> Rect2f;

/*
*	@Brief: Translate a rotated rectangle
*/
template<typename T>
cv::RotatedRect operator+(const cv::RotatedRect& rec, const cv::Point_<T>& pt)
{
	return cv::RotatedRect(cv::Point2f(rec.center.x + pt.x, rec.center.y + pt.y), rec.size, rec.angle);
}

}


namespace cvex
{
using namespace cv;

const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar MAGENTA(255, 0, 255);
const cv::Scalar YELLOW(0, 255, 255);
const cv::Scalar CYAN(255, 255, 55); ///<青色
const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar PURPLE(230, 140, 175);
const cv::Scalar PINK(203, 192, 255);
const cv::Scalar MIKU(187, 197, 57);


/*
*	@Brief: Get the euclidean distacne of two points
*/
template<typename T>
float distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
	return std::sqrt(std::pow((pt1.x - pt2.x), 2) + std::pow((pt1.y - pt2.y), 2));
}

/*
*	@Brief: Get manhattan distance of two points, i.e., |△x+△y|.
*/
template<typename T>
float distanceManhattan(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
	return std::abs(pt1.x - pt2.x) + std::abs(pt1.y - pt2.y);
}

/*
*	@Brief: get the cross point of two lines
*/
template<typename ValType>
const cv::Point2f crossPointOf(const std::array<cv::Point_<ValType>, 2>& line1, const std::array<cv::Point_<ValType>, 2>& line2)
{
	ValType a1 = line1[0].y - line1[1].y;
	ValType b1 = line1[1].x - line1[0].x;
	ValType c1 = line1[0].x*line1[1].y - line1[1].x*line1[0].y;

	ValType a2 = line2[0].y - line2[1].y;
	ValType b2 = line2[1].x - line2[0].x;
	ValType c2 = line2[0].x*line2[1].y - line2[1].x*line2[0].y;

	ValType d = a1 * b2 - a2 * b1;

	if(d == 0.0)
	{
		return cv::Point2f(FLT_MAX, FLT_MAX);
	}
	else
	{
		return cv::Point2f(float(b1*c2 - b2 * c1) / d, float(c1*a2 - c2 * a1) / d);
	}
}

inline const cv::Point2f crossPointOf(const cv::Vec4f& line1, const cv::Vec4f& line2)
{
	const std::array<cv::Point2f, 2> line1_{cv::Point2f(line1[2],line1[3]),cv::Point2f(line1[2] + line1[0],line1[3] + line1[1])};
	const std::array<cv::Point2f, 2> line2_{cv::Point2f(line2[2],line2[3]),cv::Point2f(line2[2] + line2[0],line2[3] + line2[1])};
	return crossPointOf(line1_, line2_);
}


template<typename T1, typename T2>
const cv::Rect_<T1> scaleRect(const cv::Rect_<T1>& rect, const cv::Vec<T2,2> scale, cv::Point_<T1> anchor = cv::Point_<T1>(-1, -1))
{
	if(anchor == cv::Point_<T1>(-1, -1))
	{
		anchor = cv::Point_<T1>(rect.x + rect.width / 2, rect.y + rect.height / 2);
	}
	cv::Point_<T1> tl((cv::Vec<T1, 2>(rect.tl()) - cv::Vec<T1, 2>(anchor)).mul(scale) + cv::Vec<T1, 2>(anchor));
	cv::Size_<T1> size(rect.width*scale[0], rect.height*scale[1]);
	return cv::Rect_<T1>(tl, size);
}

inline const bool isPointInRect(const Point p, const Rect rect)
{
    if (p.x < rect.tl().x || p.x > rect.br().x
            || p.y < rect.tl().y || p.y > rect.br().y)
        return false;
    else return true;
}

}
