# Outguesser
Hide, recover and detect text hidden in digital images using the Outguess algorithm on Android.

Project started in Stanford's EE368/CS232 (Digital Image Processing) by Dominique Piens and Nathan Staffa.
Implements steganography and steganalysis in Java using OpenCV. An Android app offers full functionality,
or the project can be run on your desktop's JVM.

DESCRIPTION. Encodes a text message in an image using a password. The message is encoded in the least
significant bits of the image's discrete cosine tranform (DCT) coefficients. The password is required
to decode the hidden message from the image. A detector was implemented to predict whether an image
has a hidden message. There are some minor known issues described in the links below. This project
is intended to be used on a mobile platform to quickly encode and decode short messages, and to
determine whether an image should be moved to a computer for further analysis (e.g. decode a message
when the password isn't known). Functionality for generating a training set of images with hidden
messages is also included  (used to design detector).

Projet is described graphically and in more detail at the following links:

[Poster] (http://web.stanford.edu/class/ee368/Project_Autumn_1617/Posters/poster_piens_staffa.pdf)

[Report] (http://web.stanford.edu/class/ee368/Project_Autumn_1617/Reports/report_piens_staffa.pdf)
