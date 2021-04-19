#include "utility.hpp"
#include <fstream>
#include <vector>
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

namespace utility{
	Image_info::Image_info(){
		filename_      = "_"     ;
		label_         = "others";
		numeric_label_ = 0       ;
	}

	Image_info::Image_info(string label, string filename, long num){
		filename_      = filename;
		label_         = label   ;
		numeric_label_ = num     ;
	}

	string Image_info::get_filename(){
		return filename_;
	}

	string Image_info::get_label(){
		return label_;
	}

	long Image_info::get_numeric_label(){
		return numeric_label_;
	}

	void Image_info::set_label(string label){
		label_ = label;
	}

	void Image_info::set_numeric_label(long n){
		numeric_label_ = n;
	}

	void Image_info::set_filename(string& fn){
		filename_  = fn;
	}

	rectangle random_crop(
			const matrix<rgb_pixel>& img,
			dlib::rand& rnd
			){

		/* Random Size */
		double mins = 0.466666666, maxs = 0.875;
		auto scale = mins + rnd.get_random_double()*(maxs-mins);
		auto size  = scale*std::min(img.nr(), img.nc());
		// img.nr() = img->height
		// img.nc() = img->width
		rectangle rect(size, size);

		/* Randomly moves the rectangle */
		point offset(   rnd.get_random_32bit_number() % (img.nc()-rect.width() ),
				rnd.get_random_32bit_number() % (img.nr()-rect.height())
				);
		return move_rect(rect, offset);
	}

	void randomly_crop_image(
		const matrix<rgb_pixel>& img,
		matrix<rgb_pixel>& crop     ,
		dlib::rand& rnd
		){
		/* Randomly Crops and Augments an image */

		auto rect = random_crop(img, rnd);
		extract_image_chip(img, chip_details(rect, chip_dims(227, 227)), crop);

		// Image Augmentation
		if(rnd.get_random_double() > 0.5){
			crop = fliplr(crop);
		}
		apply_random_color_offset(crop, rnd);
	}

	void crop_image_dimension(
		const matrix<rgb_pixel>& img      ,
		matrix<rgb_pixel>&       crop     ,
		int                      dimension,
		dlib::rand&              rnd
		){
		auto rect = random_crop(img, rnd);
		extract_image_chip(img, chip_details(rect, chip_dims(dimension, dimension)), crop);

		if(rnd.get_random_double() > 0.5){
			crop = fliplr(crop);
		}
		apply_random_color_offset(crop, rnd);
	}

	void randomly_crop_images(
			const matrix<rgb_pixel>& img,
			dlib::array<matrix<rgb_pixel>>& crops,
			dlib::rand& rnd,
			long num_crops
			){
		/* Randomly Crops and Augments multiple images */

		std::vector<chip_details> dets;
		// chip_details: Describes where an image chip is to be extracted from
		//		 within another image.
		for(long i=0; i<num_crops; ++i){
			auto rect = random_crop(img, rnd);
			dets.push_back(chip_details(rect, chip_dims(227, 227)));
		}

		extract_image_chips(img, dets, crops);

		/* Image Augmentation */
		for(auto&& img : crops){
			if(rnd.get_random_double() > 0.5){
				img = fliplr(img);
			}

			apply_random_color_offset(img, rnd);
		}
	}

	std::vector<Image_info> get_imagenet_listing(
		const string& root_directory ,
		const string& image_path_file,
		const string& label_file
	){
		std::cout << "Getting Imagenet Listing" << std::endl;

		ifstream label(label_file)          ;
		ifstream image_path(image_path_file);
		string   label_line, path           ;
		string   filename                   ;
		string   previous_label = ""        ;

		std::vector<Image_info> results;
		long numeric_label = -1        ;

		while(getline(label, label_line) && getline(image_path, path)){
			filename = root_directory + "/" + path;
			if(!file_exists(filename)){
				cout << "File: <"  << filename << "> does not exist." << endl;
			}
			if(label_line != previous_label){
				++numeric_label            ;
				previous_label = label_line;
			}

			Image_info temp(label_line, filename, numeric_label);
			results.push_back(temp);
		}

		return results;
	}

	void get_imagenet_dataset(
		const string& image_path_file,
		const string& label_path_file,
		std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
		std::vector<unsigned long>&                 labels
	){
		std::cout << "Fetching image dataset" << std::endl;

		ifstream label_path(label_path_file);
		ifstream image_path(image_path_file);

		string label, path              ;
		string previous_label       = "";
		unsigned long numeric_label = -1;

		dlib::rand rnd(time(0));
		while(getline(label_path, label) && getline(image_path, path)){
			dlib::matrix<dlib::rgb_pixel> img;
			while(true){
				try{
					dlib::load_image(img, path);
					crop_image_dimension(img, img, 224, rnd);
					break;
				}catch(...){
					continue;
				}
			}

			if(label != previous_label){
				++numeric_label;
				previous_label = label;
			}

			images.push_back(img)  ;
			labels.push_back(numeric_label);
		}
	}
}
