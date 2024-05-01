import math
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from st_on_hover_tabs import on_hover_tabs
import json
from streamlit_lottie import st_lottie
import random
# from io import BytesIO

import streamlit as st
import cv2
import numpy as np
from pose_detection_classification import detectPose, classifyPose,detectPose2

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Set page title
st.set_page_config(page_title="Pose Detection and Classification", layout="wide", page_icon=":man-raising-hand:", 
                   initial_sidebar_state="expanded")

# Loading the Lottie files for animations
def load_lottie_file(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
lottie_file1 =load_lottie_file('./yoga.json')
lottie_file2 =load_lottie_file('./DASH.json')
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st_lottie(lottie_file1,speed=0.8,reverse=False,height=200,width=300)
    tabs = on_hover_tabs(tabName=['Dashboard','upload pose','live Interaction','Extra Activities'], 
                         iconName=['window','image','videocam','self_improvement'], default_choice=0,
                         styles={'navtab': {'background-color':'#dde6ed',
                                            'color': '#1593af',
                                            'font-size': '18px',
                                            'transition': '.3s',
                                            'white-space': 'nowrap',
                                            'text-transform': 'uppercase'},
                                 'tabOptionsStyle': {':hover :hover': {'color': '#004280',
                                                                     'cursor': 'pointer'}},
                             },
    )
if tabs == 'Dashboard':
    # Set app title and header text with custom styling
    st.title(':red[Pose Detection and Classification]')
    c1,c2 = st.columns([0.5,0.5])
    with c1:
        st_lottie(lottie_file2,speed=0.8,reverse=False,height=500,width=500)
    with c2:
        st.info('\n\n\n\n\n\n\n\n #### Human Pose Detection And Estimation\n To analyze human body poses in images or videos.\nMediaPipe provides a robust solution capable of predicting thirty-three 3D landmarks on a human body.in real-time with high accuracy even on CPU .\n It utilises a two-step machine learning pipeline, by using a detector it first localises the person within the frame and then uses the pose landmarks detector to predict the landmarks within the region of interest.For the videos, the detector is used only for the very first frame and then the ROI is derived from the previous frame’s pose landmarks using a tracking method. \n\n\n\n\n\n\n\n\n')
if tabs == 'upload pose':
    st.title(':red[Upload an Image]')
    # File uploader for image input with centered styling
    uploaded_file = st.file_uploader("Upload an image here", type=["jpg", "png", "jpeg"], 
                                    help="Supported formats: jpg, png, jpeg",
                                    accept_multiple_files=False)
    if uploaded_file is not None:
        # # Initializing mediapipe pose class.
        # mp_pose = mp.solutions.pose

        # # Setting up the Pose function.
        # pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

        # # Initializing mediapipe drawing class, useful for annotation.
        # mp_drawing = mp.solutions.drawing_utils
        # Display the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        st.subheader('Uploaded Image')
        # image = cv2.imread('media/Tpose1.jpg')
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform pose detection and classification
        output_image, landmarks = detectPose(image=image,display=False)
        if landmarks:
            output_image, label = classifyPose(landmarks, output_image)
            # predictions.append(label)
            st.subheader('Output Image with Pose Classification')
            st.image(output_image, caption='Output Image with Pose Classification', use_column_width=True)
            st.success(f'### :green[Pose Classification: {label}]')
        else:
            st.error('No pose landmarks detected.')
    # Checkbox for enabling live video feed with custom styling
    # live_video = st.toggle("Enable Live Video", help="Start live video feed from webcam")

# Ground truth data for accuracy calculation (e.g., manually annotated pose labels)
# ground_truth_data = {
#     "./media/warriorIIpose.jpg": "Warrior II Pose",
#     "./media/treepose1.jpg": "Tree Pose",
#     "./media/Tpose1.jpg": "T Pose",
    
#     # Add more ground truth data as needed
# }

# # Initialize variables for accuracy calculation
# total_samples = len(ground_truth_data)
# correct_predictions = 0
# predictions = []

elif tabs=='live Interaction':
            st.title(':red[Interact live with the Camera]')
    # if live_video:
        # if opt=="take photo":
            img=st.camera_input("camera Input")
            if img is not None:
                
                # st.image(img)
                image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            
                st.subheader('Uploaded Image')
                # image = cv2.imread('media/Tpose1.jpg')
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Perform pose detection and classification
                output_image, landmarks = detectPose(image=image,display=False)
                if landmarks:
                    output_image, label = classifyPose(landmarks, output_image)
                    # predictions.append(label)
                    st.subheader('Output Image with Pose Classification')
                    st.image(output_image, caption='Output Image with Pose Classification', use_column_width=True)
                    st.write('Pose Classification:', label)
                else:
                    st.error('No pose landmarks detected.')
            opt=st.radio("select",["take photo","take video"],horizontal=True)
            if opt=="take video":
            
                # Setup Pose function for video.
                pose_video = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

                # Initialize the VideoCapture object to read from the webcam.
                camera_video = cv2.VideoCapture(0)
                camera_video.set(3,1280)
                camera_video.set(4,960)

                # Initialize a resizable window.
                st.title('Pose Classification')

                # Create a placeholder for the video stream
                video_placeholder = st.empty()

                # Iterate until the webcam is accessed successfully.
                while camera_video.isOpened():
                    
                    # Read a frame.
                    ok, frame = camera_video.read()
                    
                    # Check if frame is not read properly.
                    if not ok:
                        
                        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
                        continue
                    
                    # Flip the frame horizontally for natural (selfie-view) visualization.
                    frame = cv2.flip(frame, 1)
                    
                    # Get the width and height of the frame
                    frame_height, frame_width, _ =  frame.shape
                    
                    # Resize the frame while keeping the aspect ratio.
                    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
                    
                    # Perform Pose landmark detection.
                    # Replace detectPose function with the actual function you have for pose detection.
                    frame, landmarks = detectPose2(frame, pose_video, display=False)
                    
                    # Check if the landmarks are detected.
                    if landmarks:
                        
                        # Perform the Pose Classification.
                        # Replace classifyPose function with the actual function you have for pose classification.
                        frame, _ = classifyPose(landmarks, frame, display=False)
                    
                    # Display the frame.
                    video_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # Wait until a key is pressed.
                    # Retrieve the ASCII code of the key pressed
                    k = cv2.waitKey(1) & 0xFF
                    
                    # Check if 'ESC' is pressed.
                    if(k == 27):
                        
                        # Break the loop.
                        break

                # Release the VideoCapture object and close the windows.
                camera_video.release()
                cv2.destroyAllWindows()
elif tabs == 'Extra Activities':
    st.title(':red[Lets go through some extra activities like Mindfulness and Meditation]')
    # Define sections for mindfulness meditation and breathing exercises

    # Section for breathing exercises
    st.header("Breathing Exercises")
    st.write("Practice breathing exercises to reduce stress and promote calmness.")
    breathing_techniques = ["4-7-8 Breathing", "Box Breathing", "Alternate Nostril Breathing"]
    selected_breathing_technique = st.selectbox("Select a breathing technique:", breathing_techniques)
    # Display instructions for selected breathing technique
    if selected_breathing_technique == "4-7-8 Breathing":
        st.write("Inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds. Repeat for several cycles.")
    elif selected_breathing_technique == "Box Breathing":
        st.write("Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds, hold for 4 seconds. Repeat for several cycles.")
    elif selected_breathing_technique == "Alternate Nostril Breathing":
        st.write("Close your right nostril with your thumb, inhale through your left nostril. Close your left nostril with your ring finger, exhale through your right nostril. Repeat, alternating nostrils.")  
    # Add a markdown section to provide instructions and tutorial links
    st.markdown("## Interactive Tutorials")
    st.markdown("Explore the following resources to learn proper form and technique for different poses:")

    # # Create hyperlinks to tutorial resources
    # st.markdown("- [Yoga Journal: Pose Library](https://www.yogajournal.com/poses/)")
    # st.markdown("- [YouTube: Yoga With Adriene](https://www.youtube.com/user/yogawithadriene)")
    # st.markdown("- [Instagram: Yoga Poses](https://www.instagram.com/yoga/?hl=en)")

    # Create hyperlinks to tutorial resources with icons
    st.markdown("[Yoga Journal: Pose Library]<a href='https://www.yogajournal.com/poses/' target='_blank'><img src='https://img.icons8.com/color/48/000000/yoga.png' style='display:inline-block'></a>", unsafe_allow_html=True)
    st.markdown("[YouTube: Yoga With Adriene]<a href='https://www.youtube.com/user/yogawithadriene' target='_blank'><img src='https://img.icons8.com/color/48/000000/youtube-play.png' style='display:inline-block'></a>", unsafe_allow_html=True)
    st.markdown("[Instagram: Yoga Poses]<a href='https://www.instagram.com/yoga/?hl=en' target='_blank'><img src='https://img.icons8.com/fluency/48/000000/instagram-new.png' style='display:inline-block'></a>", unsafe_allow_html=True)
    
    st.title(':red[Injury Prevention Tips]')
    
    # Add text descriptions of injury prevention tips
    st.header("Warm-Up Exercises")
    st.write("Performing proper warm-up exercises before a yoga session can help prevent injuries. "
            "Include dynamic stretches, joint rotations, and gentle movements to increase blood flow "
            "to the muscles and prepare the body for physical activity.")
            
    st.header("Common Injury Prevention Techniques")
    st.write("1. Listen to your body: Pay attention to any discomfort or pain during yoga poses and "
                "modify or skip poses that cause strain or discomfort.")
    st.write("2. Use props: Props such as blocks, straps, and bolsters can assist in achieving proper "
                "alignment and reduce the risk of overstretching or straining muscles.")
            
        # Add images or videos demonstrating warm-up exercises and injury prevention techniques
    st.header("Warm-Up Exercise Demonstration")
    st.markdown("[Watch This on Youtube]<a href='https://www.youtube.com/watch?v=vWiu6ayDo2A' target='_blank'><img src='https://img.icons8.com/color/48/000000/youtube-play.png' style='display:inline-block'></a>", unsafe_allow_html=True)

        # Add recovery strategies and relaxation techniques
    st.header(':red[Recovery Strategies]')
    st.write("After a yoga session, incorporate cooldown stretches and relaxation techniques to "
                "promote muscle recovery and reduce tension. Practice deep breathing exercises, meditation, "
                "and gentle stretches to relax the body and mind.")
    st.markdown("[Watch This on Youtube]<a href='https://www.youtube.com/watch?v=CU9C3A19chA' target='_blank'><img src='https://img.icons8.com/color/48/000000/youtube-play.png' style='display:inline-block'></a>", unsafe_allow_html=True)


def main():
    # st.title("Offline Yoga Companion")
    pass
if __name__ == "__main__":
    main()
        
# # Define a list of seasonal challenges with their descriptions
# seasonal_challenges = {
#     "Halloween Yoga Challenge": "Join our Halloween-themed yoga challenge to practice spooky poses and embrace the spirit of the season!",
#     "Winter Solstice Meditation Series": "Experience peace and tranquility with our Winter Solstice meditation series, designed to help you find inner balance during the colder months.",
#     "Summer Yoga Retreat": "Escape to a virtual yoga retreat this summer and rejuvenate your mind, body, and soul with daily yoga sessions and guided meditations."
# }



    # Section for guided meditation
    # Path to your audio file (replace "path/to/your/audio/file.mp3" with the actual file path)
    # audio_path = "/Users/zohaibakhtar/Desktop/Project/CODES/Introduction to Pose Detection and Pose Classification copy/media/a.mp3"
    # st.header("Guided Meditation")
    # st.write("Listen to a guided meditation session to promote relaxation and mindfulness.")


    # # Main application code
    # def main():
    #     st.sidebar.title("Navigation")
    #     page = st.sidebar.radio("Go to", ["Home", "Injury Prevention Tips"])

    #     if page == "Home":
    #         st.title("Yoga Application")
    #         st.write("Welcome to our Yoga Application. Explore different features and resources to enhance your yoga practice.")

    #     elif page == "Injury Prevention Tips":
    #         display_injury_prevention_tips()

    # if __name__ == "__main__":
    #     main()


    # Create social sharing buttons using HTML and Markdown
# st.markdown("""
#     <h3>Share Your Progress</h3>
#     <a href="https://www.facebook.com/sharer/sharer.php?u=YOUR_URL_HERE" target="_blank">
#         <img src="https://img.icons8.com/color/48/000000/facebook.png"/>
#     </a>
#     <a href="https://twitter.com/intent/tweet?url=YOUR_URL_HERE" target="_blank">
#         <img src="https://img.icons8.com/color/48/000000/twitter.png"/>
#     </a>
#     <a href="https://www.linkedin.com/shareArticle?url=YOUR_URL_HERE" target="_blank">
#         <img src="https://img.icons8.com/color/48/000000/linkedin.png"/>
#     </a>
# """, unsafe_allow_html=True)

