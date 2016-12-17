package dn.Outguesser;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import static dn.Outguesser.outguess.OutguessTest.decodeImage;
import static dn.Outguesser.utils.FileUtils.getPath;

public class decodeWindow extends AppCompatActivity {

    /** File path to the most recently chosen image by user */
    private String PICKED_PIC_PATH = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_decode_window);
    }

    // When user clicks choose image.
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
    }

    public void onClickRun(View view) {
        if(validateInputs()) {
            runDecode();
        };
    }

    private void changeImgButton(String str) {
        ((Button)findViewById (R.id.imChooseDec)).setText (str);
    }

    private void changeDecodeButton(String str) {
        ((Button)findViewById (R.id.decodeButton)).setText (str);
    }

    /** Checks to make sure image selected and password entered.
     * Also displays snackbar. (If I get there)
     *
     * @return true if valid, false if not.
     */
    private boolean validateInputs() {
        // Get password
        EditText editText = (EditText) findViewById(R.id.edit_password);
        String password = editText.getText().toString();

        if (PICKED_PIC_PATH.isEmpty()) {
            changeDecodeButton("Choose an image first");
            return false;
        } else if (password.isEmpty()) {
            changeDecodeButton("Don't forget the password!");
            return false;
        } else { //Actually decode
            changeDecodeButton("Decoding...");
            return true;
        }
    }

    private void runDecode() {
        // Get password
        EditText editText = (EditText) findViewById(R.id.edit_password);
        String password = editText.getText().toString();

        String message = decodeImage(PICKED_PIC_PATH, password);

        // Once you get the message
        displayMessage(message);
        changeDecodeButton("Done.");
    }

    private void displayMessage(String message) {
        TextView tv = (TextView) findViewById(R.id.display_message);
        tv.setText(message);
    }



    private void onPhotoReturned(Uri photo) {
        PICKED_PIC_PATH = getPath(this, photo);
        changeImgButton("Choose another image?");

        // Update the decode button back to its original value, in case of decoding second image.
        changeDecodeButton(getString(R.string.button_startDecode));

        //Clear decoded message
        displayMessage(getString(R.string.decode_Msg));
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
