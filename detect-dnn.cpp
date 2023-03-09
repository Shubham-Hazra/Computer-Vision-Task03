#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------


int main(int argc, char** argv) try
{
    if (argc == 1 || argc == 2)
    {
        cout << "Call this program like this:" << endl;
        cout << "./detect-dnn <path_to_model> <path_to_image>" << endl;
        cout << "e.g.    ./detect-dnn ann.dat <path_to_image>" << endl;
        return 0;
    }

    if (argc > 3)
    {
        cout << "You gave too many arguments.  Just give the path to the face detector and the path to the image e.g." << endl;
        cout << "./detect-dnn ann.dat <path_to_image>" << endl;
        return 0;
    }


    net_type net;
    deserialize(argv[1]) >> net;  

    image_window win;
    for (int i = 2; i < argc; ++i)
    {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);

        // Upsampling the image will allow us to detect smaller faces but will cause the
        // program to use more RAM and run longer.
        // while(img.size() < 1800*1800)
        //     pyramid_up(img);

        // Note that you can process a bunch of images in a std::vector at once and it runs
        // much faster, since this will form mini-batches of images and therefore get
        // better parallelism out of your GPU hardware.  However, all the images must be
        // the same size.  To avoid this requirement on images being the same size we
        // process them individually in this example.
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        for (int i = 0; i < dets.size(); ++i)
        {
            draw_rectangle(img, dets[i], rgb_pixel(255,0,0),10);
        }
        save_jpeg(img, "detected.jpg");
        cout << "Hit any key to exit." << endl;
        cin.get();
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}


