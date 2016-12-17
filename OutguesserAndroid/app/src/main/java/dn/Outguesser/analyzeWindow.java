package dn.Outguesser;

import android.Manifest;
import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Parcelable;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.opencv.ml.CvSVM;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static dn.Outguesser.outguess.OutguessTest.getSteganalysisConfidence;
import static dn.Outguesser.outguess.OutguessTest.steganalyze;
import static dn.Outguesser.utils.FileUtils.getPath;

public class analyzeWindow extends AppCompatActivity {

    /** File path to the most recently chosen image by user */
    private String PICKED_PIC_PATH = "";
    private CvSVM svmMdl = new CvSVM();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_analyze_window);

        // VERY dirty way of getting the SVM model loaded in android.
        // TODO: CLEAN

        String filename = "svmFinal.yml";
        File file = new File(getCacheDir() + "/" + filename);
//        filename
        if (!file.exists())
            try {

                InputStream is = getAssets().open(filename);
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();

                FileOutputStream fos = new FileOutputStream(file);

                fos.write(buffer);
                fos.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        if (file.exists()) {
            svmMdl.load(file.getAbsolutePath());
//            displayResults("It worked!");
    }



    }

    public void onClickChooseImg(View view) {

        // Only deal with new permissions scheme if api >= 23
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP_MR1) {
            int REQUEST_CODE_ASK_PERMISSIONS = 123;
            int hasWriteExtStoragePermission = checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE);
            if (hasWriteExtStoragePermission != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        REQUEST_CODE_ASK_PERMISSIONS);
                return;
            }
        }
        // Get image
        openImageIntent();

        //TODO: If we want and have time, could display it in an image fragment maybe
    }

    public void onClickRun(View view) {
        if(validateInputs()) {
            runDetect();
        };
    }

    private void changeImgButton(String str) {
        ((Button)findViewById (R.id.imChooseAna)).setText (str);
    }

    private void changeDetectButton(String str) {
        ((Button)findViewById (R.id.detectButton)).setText (str);
    }

    private void displayResults(String message) {
        TextView tv = (TextView) findViewById(R.id.analysisResults);
        tv.setText(message);
    }

    /** Checks to make sure image selected.
     *
     * @return true if valid, false if not.
     */
    private boolean validateInputs() {
        if (PICKED_PIC_PATH.isEmpty()) {
            changeDetectButton("Choose an image first");
            return false;
        } else { //Actually detect
            changeDetectButton("Running steganalysis...");
            return true;
        }
    }

    private void runDetect() {
//        String message = decodeImage(PICKED_PIC_PATH, password);
        double measure = steganalyze(svmMdl, PICKED_PIC_PATH);
        double confidence = getSteganalysisConfidence(measure);

        if (measure < 0) { // Negative result
            displayResults("Our model is " + Double.toString(confidence*100) +
                    "% sure this image is NOT encoded with OutGuess.");

        } else {  // Positive result: probably steg'd.
            displayResults("Our model is " + Double.toString(confidence*100) +
                    "% sure this image IS encoded with OutGuess. Gasp!");
        }



//        displayResults(message);
//        changeDecodeButton("Done.");
    }



    private void onPhotoReturned(Uri photo) {
        PICKED_PIC_PATH = getPath(this, photo);
        changeImgButton("Choose another image?");

        // Update the detect button back to its original value, in case of second image.
        changeDetectButton(getString(R.string.button_startDetect));

        //Clear detection results
        displayResults("");
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // for handling the picture gettingness.
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE_REQUEST_CODE) {
                final boolean isCamera;
                if (data == null || (data.getData() == null && data.getClipData() == null)) {
                    isCamera = true;
                } else {
                    final String action = data.getAction();
                    if (action == null) {
                        isCamera = false;
                    } else {
                        isCamera = action.equals(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    }
                }

                Uri selectedImageUri;
                if (isCamera) {
                    selectedImageUri = outputFileUri;
                } else {
                    selectedImageUri = data == null ? null : data.getData();
                }
                onPhotoReturned(selectedImageUri);
            }
        }

    }

    /* This handles the opening of the file or capturing from camera */
    private Uri outputFileUri;
    private int SELECT_PICTURE_REQUEST_CODE = 1337;

    private void openImageIntent() {

        // Filesystem.
        final Intent galleryIntent = new Intent();
        galleryIntent.setType("image/*");
        galleryIntent.setAction(Intent.ACTION_PICK);

        // Chooser of filesystem options.
        final Intent chooserIntent = Intent.createChooser(galleryIntent, "Select Source");

        startActivityForResult(chooserIntent, SELECT_PICTURE_REQUEST_CODE);
    }


}
