package dn.Outguesser;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import org.opencv.android.OpenCVLoader;


public class MainActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

    }

    public void onEncodeClick(View view) {
        // Launch intent for encode window.
        Intent intent = new Intent(this, encodeWindow.class);
        startActivity(intent);
    }

    public void onDecodeClick(View view) {
        // Launch intent for decode window.
        Intent intent = new Intent(this, decodeWindow.class);
        startActivity(intent);
    }

    public void onDetectClick(View view) {
        // for steganalysis
        Intent intent = new Intent(this, analyzeWindow.class);
        startActivity(intent);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);


    }

}
