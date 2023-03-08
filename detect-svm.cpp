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
        // In this example we are going to train a face detector based on the
        // small faces dataset in the examples/faces directory.  So the first
        // thing we do is load that dataset.  This means you need to supply the
        // path to this faces folder as a command line argument so we will know
        // where it is.
        if (argc != 3)
        {
            cout << "Give the path to the model and path to the image as the argument to this" << endl;
            cout << "   ./dnn_mmod_ex <path_to_svm_model> <path_to_image>" << endl;
            cout << "e.g.   ./dnn_mmod_ex masked01.svm <path_to_image>" << endl;
            cout << endl;
            return 0;
        }
        const std::string svm_detector = argv[1];
        const std::string image_path = argv[2];

        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
        object_detector<image_scanner_type> detector;
        deserialize(svm_detector) >> detector;
        
        matrix<rgb_pixel> img;
        load_image(img, image_path);

        // Upsampling the image will allow us to detect smaller faces but will cause the
        // program to use more RAM and run longer.
        while(img.size() < 1800*1800)
            pyramid_up(img);

        image_window win; 
        std::vector<rectangle> dets = detector(img);
        win.clear_overlay();
        win.set_image(img);
        win.add_overlay(dets, rgb_pixel(255,0,0));
        cout << "Hit any key to exit." << endl;
        cin.get();
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}