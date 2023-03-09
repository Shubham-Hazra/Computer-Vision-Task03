#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>

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
            cout << "Give the path to the model and path to the image as the argument to this" << endl;
            cout << "   ./detect-svm <path_to_svm_model> <path_to_image>" << endl;
            cout << "e.g.   ./detect-svm masked01.svm <path_to_image>" << endl;
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
        for (int i = 0; i < dets.size(); ++i)
        {
            draw_rectangle(img, dets[i], rgb_pixel(255,0,0),10);
        }
        save_jpeg(img, "detected.jpg");
        cout << "Hit any key to exit." << endl;
        cin.get();
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}
