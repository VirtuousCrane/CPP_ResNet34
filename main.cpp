#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include "src/utility.hpp"

using namespace std    ;
using namespace dlib   ;
using namespace utility;

const unsigned int  MINIBATCH_SIZE              = 16  ;
const unsigned int  NO_OF_CLASSES               = 10  ;
const unsigned int  SYNC_DURATION               = 60  ;
const unsigned int  IMAGE_SIZE                  = 224 ;
const unsigned long ITERATIONS_WITHOUT_PROGRESS = 2000;
const double        INITIAL_LEARNING_RATE       = 0.1 ;


// TODO: Replace
const string ROOT_DIR = "";

const string TRAINING_IMAGE_PATH = "";
const string TRAINING_LABEL_PATH = "";
const string TESTING_IMAGE_PATH  = "";
const string TESTING_LABEL_PATH  = "";

const string TRAINING_IMAGE_ROOT = "";
const string TESTING_IMAGE_ROOT  = "";

const string SYNC_FILE     = "";

const string NETWORK_PATH = "";

// ======================================================================
//  Defining The Neural Network (ResNet 34)
// ======================================================================

// # First Layer
template<
	template<typename> class BN,
	template<typename> class ACTIVATION,
	typename                 SUBNET
	>
using custom_ResNet_input_layer = ACTIVATION<BN<con<64, 7, 7, 2, 2, SUBNET>>>;

template<typename SUBNET> using relu_input_layer  = custom_ResNet_input_layer<bn_con, relu, SUBNET>;
template<typename SUBNET> using arelu_input_layer = custom_ResNet_input_layer<affine, relu, SUBNET>;

// # First Layer Max Pool
template<typename SUBNET> using input_pooling = max_pool<3, 3, 2, 2, SUBNET>;

// # Resnet Building Block
// conv -> BN -> ACTIVATION -> conv -> BN
template<
	int N                              ,
	template<typename> class BN        ,
	template<typename> class ACTIVATION,
	int stride                         ,
	typename SUBNET
	>
using residual_block_not_activated =    add_prev1<
					BN<
					con<N, 3, 3, stride, stride,

					ACTIVATION<
					BN<
					con<N, 3, 3, stride, stride,
					tag1<SUBNET>
					>>>>>>;

template<int N, typename SUBNET> using relu_block  = relu<residual_block_not_activated<N, bn_con, relu, 1, SUBNET>>;
template<int N, typename SUBNET> using arelu_block = relu<residual_block_not_activated<N, bn_con, relu, 1, SUBNET>>;

template<typename SUBNET> using block_64  = relu_block<64 , SUBNET>;
template<typename SUBNET> using block_128 = relu_block<128, SUBNET>;
template<typename SUBNET> using block_256 = relu_block<256, SUBNET>;
template<typename SUBNET> using block_512 = relu_block<512, SUBNET>;

template<typename SUBNET> using ablock_64  = arelu_block<64 , SUBNET>;
template<typename SUBNET> using ablock_128 = arelu_block<128, SUBNET>;
template<typename SUBNET> using ablock_256 = arelu_block<256, SUBNET>;
template<typename SUBNET> using ablock_512 = arelu_block<512, SUBNET>;

// # Resnet Downsampling block
// conv -> BN -> ACTIVATION -> conv -> BN
// shortcut = 1x1 conv
template<
	int N,
	template <typename> class BN,
	template <typename> class ACTIVATION,
	typename SUBNET
	>
using residual_downsampling_not_activated = add_prev2<

						con<N, 1, 1, 2, 2,
						skip1<

						tag2<
						residual_block_not_activated<N, BN, ACTIVATION, 2,

						tag1<SUBNET>
						>>>>>;

template<int N, typename SUBNET> using relu_downsampling  = relu<residual_downsampling_not_activated<N, bn_con, relu, SUBNET>>;
template<int N, typename SUBNET> using arelu_downsampling = relu<residual_downsampling_not_activated<N, affine, relu, SUBNET>>;

template<typename SUBNET> using downsampling_128 = relu_downsampling<128, SUBNET>;
template<typename SUBNET> using downsampling_256 = relu_downsampling<256, SUBNET>;
template<typename SUBNET> using downsampling_512 = relu_downsampling<512, SUBNET>;

template<typename SUBNET> using adownsampling_128 = arelu_downsampling<128, SUBNET>;
template<typename SUBNET> using adownsampling_256 = arelu_downsampling<256, SUBNET>;
template<typename SUBNET> using adownsampling_512 = arelu_downsampling<512, SUBNET>;

using net_type = loss_multiclass_log<
			fc<NO_OF_CLASSES,
			avg_pool_everything<

			repeat<2,
			block_512,
			downsampling_512<

			repeat<5,
			block_256,
			downsampling_256<

			repeat<3,
			block_128,
			downsampling_128<

			repeat<3,
			block_64,

			input_pooling<
			relu_input_layer<
			input_rgb_image_sized<IMAGE_SIZE>
			>>>>>>>>>>>>;

using tnet = loss_multiclass_log<
			fc<NO_OF_CLASSES,
			avg_pool_everything<

			repeat<2,
			ablock_512,
			adownsampling_512<

			repeat<5,
			ablock_256,
			adownsampling_256<

			repeat<3,
			ablock_128,
			adownsampling_128<

			repeat<3,
			ablock_64,

			input_pooling<
			relu_input_layer<
			input_rgb_image_sized<IMAGE_SIZE>
			>>>>>>>>>>>>;
// ======================================================================

int main(int argc, char** argv) try{
// ======================================================================
//  Loading The Dataset
// ======================================================================
	std::vector<matrix<rgb_pixel>> training_images, testing_images;
	std::vector<unsigned long>     training_labels, testing_labels;

	utility::get_imagenet_dataset(
				TRAINING_IMAGE_PATH,
				TRAINING_LABEL_PATH,
				TRAINING_IMAGE_ROOT,
				training_images    ,
				training_labels
	);
	utility::get_imagenet_dataset(
				TESTING_IMAGE_PATH,
				TESTING_LABEL_PATH,
				TESTING_IMAGE_ROOT,
				testing_images    ,
				testing_labels
	);

	const auto number_of_classes = training_labels.back()+1;

	cout << "No. of image in training set: " << training_images.size() << endl;
	cout << "No. of classes: "               << number_of_classes      << endl;

// ======================================================================
//  Configuring the Neural Network
// ======================================================================

	set_dnn_prefer_smallest_algorithms();

	const double weight_decay          = 0.0001               ;
	const double momentum              = 0.9                  ;

	net_type net;
	dnn_trainer<net_type> trainer(net, sgd(weight_decay, momentum));

	trainer.be_verbose();
	trainer.set_learning_rate(INITIAL_LEARNING_RATE);
	trainer.set_synchronization_file(SYNC_FILE, std::chrono::seconds(SYNC_DURATION));
	trainer.set_iterations_without_progress_threshold(ITERATIONS_WITHOUT_PROGRESS);

// ======================================================================
//  Training The Neural Network
// ======================================================================

	std::vector<matrix<rgb_pixel>> samples;
	std::vector<unsigned long>     labels ;

	dlib::rand rnd(69420);
	while(trainer.get_learning_rate() >= INITIAL_LEARNING_RATE * 1e-3){
		samples.clear();
		labels.clear();

		while(samples.size() < MINIBATCH_SIZE){
			auto index = rnd.get_random_32bit_number() % training_images.size();
			samples.push_back(training_images[index]);
			labels.push_back(training_labels[index]);
		}
		trainer.train_one_step(samples, labels);
	}

	trainer.get_net();
	cout << "Saving Network" << endl;
	serialize(NETWORK_PATH) << net;

// ======================================================================
//  Testing The Neural Network
// ======================================================================

	deserialize(NETWORK_PATH) >> net;
	softmax<tnet::subnet_type> snet ;
	snet.subnet() = net.subnet()    ;

	cout << "Testing the Network" << endl;

	int training_num_right = 0;
	int training_num_wrong = 0;
	int testing_num_right  = 0;
	int testing_num_wrong  = 0;

	for(int i=0; i<((int)training_images.size())/4; i++){
		matrix<float, 1, NO_OF_CLASSES> p = sum_rows(mat(snet(training_images[i])));
		if(index_of_max(p) == training_labels[i]){
			++training_num_right;
		}else{
			++training_num_wrong;
		}
	}

	cout << "Training acc.: " << 100*(training_num_right/(double)(training_num_right + training_num_wrong)) << endl;

	for(int i=0; i<testing_images.size(); i++){
		matrix<float, 1, NO_OF_CLASSES> p = sum_rows(mat(snet(testing_images[i])));
		if(index_of_max(p) == testing_labels[i]){
			++testing_num_right;
		}else{
			++testing_num_wrong;
		}
	}

	cout << "Testing acc.: " << 100*(testing_num_right/(double)(testing_num_right + testing_num_wrong)) << endl;

} catch(std::exception& e){
	cout << e.what() << endl;
}
