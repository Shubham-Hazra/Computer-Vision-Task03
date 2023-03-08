#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;



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
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1 || argc == 2)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex <path_to_model> <path_to_image>" << endl;
            return 0;
        }
        if (argc > 3)
        {
            cout << "Too many arguments.  Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex <path_to_model> <path_to_image>" << endl;
            return 0;
        }
        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

        image_window win, win_faces;
        // Loop over all the images provided on the command line.
        // cout << "processing image " << argv[i] << endl;
        array2d<rgb_pixel> img;
        load_image(img, argv[2]);
        // Make the image larger so we can detect small faces.
        pyramid_up(img);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<rectangle> dets = detector(img);
        // cout << "Number of faces detected: " << dets.size() << endl;

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        std::vector<full_object_detection> shapes;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            // cout << "number of parts: "<< shape.num_parts() << endl;
            // cout << "pixel position of first part:  " << shape.part(0) << endl;
            // cout << "pixel position of second part: " << shape.part(1) << endl;
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            shapes.push_back(shape);
        }

        // Now let's view our face poses on the screen.
        win.clear_overlay();
        win.set_image(img);
        win.add_overlay(render_masked_face_detections(shapes));

        // // We can also extract copies of each face that are cropped, rotated upright,
        // // and scaled to a standard size as shown here:
        // dlib::array<array2d<rgb_pixel> > face_chips;
        // extract_image_chips(img, get_face_chip_details(shapes), face_chips);
        // win_faces.set_image(tile_images(face_chips));

        cout << "Hit any key to exit." << endl;
        cin.get();
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

