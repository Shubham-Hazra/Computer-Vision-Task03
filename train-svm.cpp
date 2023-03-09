#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  

    try
    {
        if (argc != 3)
        {
            cout << "Give the path to the directory and path to training.xml as the argument to this" << endl;
            cout << "   ./train-svm <path_to_directory> <path_to_training.xml>" << endl;
            cout << endl;
            return 0;
        }
        const std::string faces_directory = argv[1];
        const std::string training_xml = argv[2];

        dlib::array<array2d<unsigned char> > images_train;

        std::vector<std::vector<rectangle> > face_boxes_train;
        
        load_image_dataset(images_train, face_boxes_train,training_xml);
        
        upsample_image_dataset<pyramid_down<2> >(images_train, face_boxes_train);
        
        add_image_left_right_flips(images_train, face_boxes_train);
        
        cout << "num training images: " << images_train.size() << endl;  
        
        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        image_scanner_type scanner;

        scanner.set_detection_window_size(80, 80); 
        structural_object_detection_trainer<image_scanner_type> trainer(scanner);

        trainer.set_num_threads(4);  

        trainer.set_c(1);
  
        trainer.be_verbose();

        trainer.set_epsilon(0.01);

        object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);

        cout << "training results: " << test_object_detection_function(detector, images_train, face_boxes_train) << endl;

        image_window hogwin(draw_fhog(detector), "Learned fHOG detector");

        serialize("face_detector.svm") << detector;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

