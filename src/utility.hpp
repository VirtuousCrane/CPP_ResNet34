#include <string>
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

#ifndef UTILITY_HPP
#define UTILITY_HPP

namespace utility{
	class Image_info{
		private:
			std::string filename_   ;
			std::string label_      ;
			long numeric_label_;
		public:
			Image_info();
			Image_info(std::string label, std::string filename, long num);

			std::string get_filename()        ;
			std::string get_label()           ;
			long get_numeric_label()          ;

			void set_label(std::string label) ;
			void set_numeric_label(long n)    ;
			void set_filename(std::string& fn);
	};

	dlib::rectangle random_crop(
			const dlib::matrix<dlib::rgb_pixel>& img,
			dlib::rand& rnd
	);

	void randomly_crop_image(
		const dlib::matrix<dlib::rgb_pixel> &img,
		dlib::matrix<dlib::rgb_pixel>& crop    ,
		dlib::rand& rnd
	);

	void crop_image_dimension(
		const dlib::matrix<dlib::rgb_pixel>& img      ,
		dlib::matrix<dlib::rgb_pixel>&       crop     ,
		int                                  dimension,
		dlib::rand&                          rnd
	);

	void randomly_crop_images(
		const dlib::matrix<dlib::rgb_pixel> &img         ,
		dlib::array<dlib::matrix<dlib::rgb_pixel>>& crops,
		dlib::rand& rnd                                  ,
		long num_crops
	);

	std::vector<Image_info> get_imagenet_listing(
		const std::string& root_directory ,
		const std::string& image_path_file,
		const std::string& label_file
	);

	void get_imagenet_dataset(
		const std::string& image_path_file               ,
		const std::string& label_path_file               ,
		std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
		std::vector<unsigned long>&                labels
	);
}

#endif
