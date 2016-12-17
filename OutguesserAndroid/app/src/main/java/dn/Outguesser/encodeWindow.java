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
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import dn.Outguesser.outguess.OutguessTest;

import static dn.Outguesser.utils.FileUtils.getPath;


/* NOTE: Crash will occur if user attempts to press "OK" after taking an image with the camera
if the orientation is not the same as when the camera was launched. This is not a hard fix,
but time is almost out!
 */
public class encodeWindow extends AppCompatActivity {

    /** File path to the most recently chosen image by user */
    private String PICKED_PIC_PATH = "";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_encode_window);

    }

    // When user clicks choose image
    public void onClickChooseImg(View view) {

        // Only deal with new permissions scheme if api >= 23
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP_MR1) {
            int REQUEST_CODE_ASK_PERMISSIONS = 123;
            int hasWriteExtStoragePermission = checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            if (hasWriteExtStoragePermission != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        REQUEST_CODE_ASK_PERMISSIONS);
                return;
            }
        }
        // Get image
        openImageIntent();
    }

    private void changeEncodeButton(String str) {
        ((Button)findViewById (R.id.encodeButton)).setText (str);
    }

    private void changeImgButton(String str) {
        ((Button)findViewById (R.id.imChooseEnc)).setText (str);
    }

    public void onClickRun(View view) {
        // Display message that encoding is in progress on button.
        changeEncodeButton("Encoding...");
        if(validateInputs()) {
            runEncode();
        };
    }

    /** Checks to make sure image selected, message entered, and password entered.
     * Also displays snackbar. (actually just changing button message)
     *
     * @return true if valid, false if not.
     */
    private boolean validateInputs() {
        // Get message
        EditText editText = (EditText) findViewById(R.id.edit_message);
        String message = editText.getText().toString();
        // Get password
        editText = (EditText) findViewById(R.id.edit_password);
        String password = editText.getText().toString();

        if (PICKED_PIC_PATH.isEmpty()) {
            changeEncodeButton("Choose an image first");
            return false;
        } else if (message.isEmpty()) {
            changeEncodeButton("Enter a message, silly");
            return false;
        } else if (password.isEmpty()) {
            changeEncodeButton("Don't forget a password!");
            return false;
        } else { //Actually encode
            return true;
        }
    }


    /** Run when user submits the message and password. */
    private void runEncode() {
        // Get filename
        EditText editText = (EditText) findViewById(R.id.edit_filename);
        String filename = editText.getText().toString();
        if (filename.isEmpty()) { // Generate a filename.
            filename = UUID.randomUUID().toString();
        }
        // Get message
        editText = (EditText) findViewById(R.id.edit_message);
        String message = editText.getText().toString().concat("\0");
        // Get password
        editText = (EditText) findViewById(R.id.edit_password);
        String password = editText.getText().toString();

        // Set output file directory. Create if it doesn't exist.
        File outFolder = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "TotallyNormalImages");
        if (!outFolder.mkdirs()) {
            Log.e("encodeWindow/runEncode", "Directory not created (maybe because it exists already)");
        }
        String pathOut = outFolder.toString().concat("/" + filename + ".png");

        // Run the encoder
        OutguessTest.encodeImage(PICKED_PIC_PATH, pathOut, message, password);

        //  Notify the mediaserver! This ensures it will be detected and added to the gallery.
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        Uri contentUri = Uri.fromFile(new File(pathOut));
        mediaScanIntent.setData(contentUri);
        this.sendBroadcast(mediaScanIntent);

        changeEncodeButton("Done! (Go Back to Decode)");
    }

    private void onPhotoReturned(Uri photo) {
        PICKED_PIC_PATH = getPath(this, photo);  // Register the selected image's path.

        changeImgButton("Choose another image?"); // Update button message

        // Update the encode button back to its original value, in case of encoding second image.
        changeEncodeButton(getString(R.string.button_startEncode));
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



    private Uri outputFileUri;
    private int SELECT_PICTURE_REQUEST_CODE = 1337;

    private void openImageIntent() {

        // Determine Uri of camera image to save.
        final File root = new File(Environment.getExternalStorageDirectory() + File.separator + "MyDir" + File.separator);
        root.mkdirs();
        final String fname = "preEncodeToDelete";
        final File sdImageMainDirectory = new File(root, fname);
        outputFileUri = Uri.fromFile(sdImageMainDirectory);

        // Camera.
        final List<Intent> cameraIntents = new ArrayList<Intent>();
        final Intent captureIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        final PackageManager packageManager = getPackageManager();
        final List<ResolveInfo> listCam = packageManager.queryIntentActivities(captureIntent, 0);
        for(ResolveInfo res : listCam) {
            final String packageName = res.activityInfo.packageName;
            final Intent intent = new Intent(captureIntent);
            intent.setComponent(new ComponentName(res.activityInfo.packageName, res.activityInfo.name));
            intent.setPackage(packageName);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, outputFileUri);
            cameraIntents.add(intent);
        }

        // Filesystem.
        final Intent galleryIntent = new Intent();
        galleryIntent.setType("image/*");
        galleryIntent.setAction(Intent.ACTION_PICK);

        // Chooser of filesystem options.
        final Intent chooserIntent = Intent.createChooser(galleryIntent, "Select Source");

        // Add the camera options.
        chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, cameraIntents.toArray(new Parcelable[cameraIntents.size()]));

        startActivityForResult(chooserIntent, SELECT_PICTURE_REQUEST_CODE);
    }





}
