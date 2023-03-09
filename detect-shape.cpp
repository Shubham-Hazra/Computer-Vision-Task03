#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include "dlib/algs.h"
#include "dlib/pixel.h"
#include "dlib/matrix.h"
#include "dlib/gui_widgets/fonts.h"
#include <cmath>

using namespace dlib;
using namespace std;

template <typename image_type>
inline void draw_masks(image_type& c,
        const std::vector<full_object_detection>& dets,
        const rgb_pixel color = rgb_pixel(0,255,0)
    )
    {
         for (unsigned long i = 0; i < dets.size(); ++i)
        {
            const full_object_detection& d = dets[i];
            for (unsigned long j = 2; j < d.num_parts() ; ++j)
            {
                draw_line(c, d.part(j-1), d.part(j), color);
            }
            draw_line(c, d.part(d.num_parts()-1), d.part(1), color);
        }
    }

inline std::vector<image_window::overlay_line> render_masked_face_detections (
        const std::vector<full_object_detection>& dets,
        const rgb_pixel color = rgb_pixel(0,255,0)
    )
    {
        std::vector<image_window::overlay_line> lines;
        for (unsigned long i = 0; i < dets.size(); ++i)
        {
            const full_object_detection& d = dets[i];

            for (unsigned long j = 2; j < d.num_parts() ; ++j)
            {
                lines.push_back(image_window::overlay_line(d.part(j-1), d.part(j), color));
            }
            lines.push_back(image_window::overlay_line(d.part(d.num_parts()-1), d.part(1), color));
        }
        return lines;
    }


// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1 || argc == 2)
        {
            cout << "Call this program like this:" << endl;
            cout << "./detect-shape <path_to_model> <path_to_image> <show>" << endl;
            cout << "The show argument is optional.  If you supply it then the program will show you the output." << endl;
            return 0;
        }
        if (argc > 4)
        {
            cout << "Too many arguments.  Call this program like this:" << endl;
            cout << "./detect-shape <path_to_model> <path_to_image> <show>" << endl;
            return 0;
        }
        ofstream file;
        file.open ("predicted.txt");
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

        image_window win, win_faces;
        matrix<rgb_pixel> img;
        load_image(img, argv[2]);
        pyramid_up(img);
        std::vector<rectangle> dets = detector(img);
        std::vector<full_object_detection> shapes;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            shapes.push_back(shape);
        }
        draw_masks(img, shapes);
        save_jpeg(img, "masked_face.jpg");
        for (unsigned long i = 0; i < shapes.size(); ++i)
        {
            file << "face " << i << endl;
            const full_object_detection& d = shapes[i];

            for (unsigned long j = 2; j < d.num_parts() ; ++j)
            {
                file << d.part(j) << endl;
            }
        }
        file.close();
        if (argv[3]=="show")
        {
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_masked_face_detections(shapes));
            cout << "Hit any key to exit." << endl;
            cin.get();
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------