package dn.Outguesser.outguess;
// EE 368 Final Project - Dominique Piens & Nathan Staffa
import org.opencv.core.*;
import org.opencv.highgui.*;
import org.opencv.imgproc.*;
import org.opencv.utils.*;
import org.opencv.ml.*;
import java.util.Random;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.ArrayList;
import java.util.TreeSet;
/*
// Uncomment if you use commented code in encodeImage to save
// images with java instead of OpenCV.
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import javax.imageio.ImageIO;
*/
public class OutguessTest {
	private static int ppb = 64; // Pixels per block.

	private static Mat lumQuant;
	private static Mat chromeQuant;

	private static boolean quantMatsInitialized = false;

	/**
	 * Quantization matrices used in @see im2dcts and @see dcts2im
	 * are initialized here. They get initialized at first use only.
	 * @see quantMatsInitialized is used to indicate this.
	 * Values drawn from JPG standard ITU CCIT T.81.
	 */
	private static void initQuantMats() {
		int[] lumVals = { 16, 11, 10, 16, 24, 40, 51, 61,
				12, 12, 14, 19, 26, 58, 60, 55,
				14, 13, 16, 24, 40, 57, 69, 56,
				14, 17, 22, 29, 51, 87, 80, 62,
				18, 22, 37, 56, 68, 109, 103, 77,
				24, 35, 55, 64, 81, 104, 113, 92,
				49, 64, 78, 87, 103, 121, 120, 101,
				72, 92, 95, 98, 112, 100, 103, 99 };
		lumQuant = new MatOfInt(lumVals);
		lumQuant = lumQuant.reshape(0, 8);
		lumQuant.convertTo(lumQuant, CvType.CV_32FC1);

		int[] chromVals = { 17, 18, 24, 47, 99, 99, 99, 99,
				18, 21, 26, 66, 99, 99, 99, 99,
				24, 26, 56, 99, 99, 99, 99, 99,
				47, 66, 99, 99, 99, 99, 99, 99,
				99, 99, 99, 99, 99, 99, 99, 99,
				99, 99, 99, 99, 99, 99, 99, 99,
				99, 99, 99, 99, 99, 99, 99, 99,
				99, 99, 99, 99, 99, 99, 99, 99 };
		chromeQuant = new MatOfInt(lumVals);
		chromeQuant = chromeQuant.reshape(0, 8);
		chromeQuant.convertTo(chromeQuant, CvType.CV_32FC1);
		quantMatsInitialized = true;
	}

	/**
	 * Used to get the binary representation of the message to encode with
	 * Outguess. Given the message as a string, returns its binary
	 * representation as ASCII characters.
	 * @param message as a string.
	 * @return input message's .
	 */
	private static String string2bin(String s) {
		byte[] msgBytes = s.getBytes(StandardCharsets.US_ASCII);
		String msgBin = "";
		for (byte b : msgBytes) {
			String nextChar = Integer.toBinaryString(b);
			// Zero pad the character representation so it is 8 bits.
			while (nextChar.length() < 8) {
				nextChar = "0" + nextChar;
			}
			msgBin += nextChar;
		}
		return msgBin;
	}

	/**
	 * Applies part of the JPG compression pipeline until DCT coefficients are
	 * obtained. Operating on YCbCr channels, computes the DCT coefficients for
	 * every 8x8 pixel block.
	 * @param BGR color image.
	 * @return image's DCT coefficients for its three channels concatenated.
	 */
	private static Mat im2dcts(Mat im) {
		// Load quantization matrices.
		if (!quantMatsInitialized) {
			initQuantMats();
		}
		int m = im.rows() / 8;
		int n = im.cols() / 8;
		int numBlocks = m*n;
		// 3 channels, 64 pixels per block.
		Mat blockDctsAll = new Mat(numBlocks * 3 * ppb, 1, CvType.CV_32FC1);

		// Change image to YCrCb color space
		Imgproc.cvtColor(im, im, Imgproc.COLOR_BGR2YCrCb);

		// Split color channels
		List<Mat> ccs = new ArrayList<Mat>();
		Core.split(im, ccs);
		// For each color channel.
		for (int ccNo = 0; ccNo < 3; ccNo++) {
			Mat quantMat;
			if (ccNo == 0) {
				quantMat = lumQuant;
			} else {
				quantMat = chromeQuant;
			}
			Mat cc = ccs.get(ccNo);
			cc.convertTo(cc, CvType.CV_32SC1);
			// Center around 0.
			Core.add(cc, new Scalar(-128), cc);
			cc.convertTo(cc, CvType.CV_32FC1);


			ArrayList<Mat> blocks = new ArrayList<Mat>();

			// Iterate over complete blocks
			for (int rowInd = 0; rowInd < m; rowInd++) {
				for (int colInd = 0; colInd < n; colInd++) {
					int iMin = rowInd * 8;
					int iMax = iMin + 8;
					int jMin = colInd * 8;
					int jMax = jMin + 8;
					blocks.add(cc.submat(iMin, iMax, jMin, jMax));
				}
			}
			// TODO Handle incomplete blocks
			// 64 pixels per block.
			int rowStart = ccNo*numBlocks*ppb;
			int rowEnd = rowStart + numBlocks*ppb;
			Mat blockDcts = blockDctsAll.submat(rowStart, rowEnd, 0, 1);

			// Get the DCT coefficiens for every block, and quantize them
			for (int blockNo = 0; blockNo < numBlocks; blockNo++) {
				int iMin = blockNo*ppb;
				int iMax = iMin + ppb;
				Mat blockDct = new Mat();
				Core.dct(blocks.get(blockNo), blockDct);

				Mat blockDctFlat = blockDcts.submat(iMin, iMax, 0, 1);
				Core.divide(blockDct, quantMat, blockDct);
				// Vectorize DCT coefficients and concatenate.
				blockDct = blockDct.reshape(0, ppb);
				blockDct.copyTo(blockDctFlat);
			}
			cc.release();
			ccs.get(ccNo).release();
		}

		// Convert to 8-bit signed ints for manipulating the LSB.
		blockDctsAll.convertTo(blockDctsAll, CvType.CV_8SC1);

		// Suggest to SVM to clean up memory since Mats are large, and JVM is
		// not aware of their actual size.
		System.gc();
		return blockDctsAll;
	}

	/**
	 * Applies part of the JPG decompression pipeline turning DCT coefficients
	 * into an image of the given size
	 * @param Quantized 8-bit signed DCT coefficients.
	 * @param Size of image to rebuild.
	 * @return image corresponding to input DCT coefficients.
	 */
	private static Mat dcts2im(Mat dctsIn, Size sz) {
		// Get quantization matrices
		if (!quantMatsInitialized) {
			initQuantMats();
		}
		int m = (int)sz.height / 8;
		int n = (int)sz.width / 8;
		int numBlocks = n*m;

		List<Mat> ccs = new ArrayList<Mat>();

		// Convert to floats for idct
		dctsIn.convertTo(dctsIn, CvType.CV_32FC1);

		// For every color channel
		for (int ccNo = 0; ccNo < 3; ccNo++) {
			Mat quantMat;
			if (ccNo == 0) {
				quantMat = lumQuant;
			} else {
				quantMat = chromeQuant;
			}
			Mat cc = Mat.zeros(m*8, n*8, CvType.CV_32FC1);

			// 64 for the number of pixels per block.
			int rowStart = ccNo*numBlocks*ppb;
			int rowEnd = rowStart + numBlocks*ppb;

			Mat dcts = dctsIn.submat(rowStart, rowEnd, 0, 1);
			int numBlocks2 = dcts.rows() / ppb;

			// Reassemble blocks into full image.
			for (int blockNo = 0; blockNo < numBlocks2; blockNo++) {
				int iMin = blockNo * ppb;
				int iMax = iMin + ppb;
				Mat blockCoeffs = dcts.submat(iMin, iMax, 0, 1);
				blockCoeffs = blockCoeffs.reshape(0, 8);
				// Undo quantization
				Core.multiply(blockCoeffs, quantMat, blockCoeffs);
				Mat block = new Mat();
				Core.idct(blockCoeffs, block);
				// Place block in final image.
				int xMin = (blockNo / n) * 8;
				int xMax = xMin + 8;
				int yMin = (blockNo % n) * 8;
				int yMax = yMin + 8;
				block.copyTo(cc.submat(xMin, xMax, yMin, yMax));
			}
			Core.add(cc, new Scalar(128), cc); // Return intensities to 0-255.
			ccs.add(cc);
		}
		// Merge color channels
		Mat im = new Mat();
		Core.merge(ccs, im);

		// Memory clean up.
		dctsIn.release();
		ccs.clear();
		System.gc();
		return im;
	}

	/**
	 * Encodes DCT coefficients with Outguess. Visits the DCT coefficients in
	 * pseudo-random order determined by pass (the seed). Changes the
	 * least significant bit at visited coefficients to match the next bit
	 * in the message.
	 * @param DCT coefficients (8-bits signed) of image to encode.
	 * @param Binary message to encode (string of 0's and 1's).
	 * @param Hashed password (int form of password).
	 * @param Maximum number of bits to encode.
	 */
	private static void outguessEncode(Mat dcts, String msg, int pass, int maxBits) {
		// Keep track of coefficients written to so we do not overwrite them.
		TreeSet<Integer> visited = new TreeSet<Integer>();
		// Set PRNG's seed
		Random rand = new Random();
		rand.setSeed(pass);

		int bitsWrit = 0;
		int bitsTotal = msg.length();
		while(bitsWrit < bitsTotal && bitsWrit < maxBits) {
			// Get next pseudo-random coefficient index.
			int ind = rand.nextInt(dcts.rows());
			// Make sure we do not overwrite a message bit.
			if (visited.contains(ind)) {
				continue;
			} else {
				visited.add(ind);
			}
			byte[] val = {0};
			dcts.get(ind, 0, val);
			// Only modify coefficients if they are neither 1 nor 0.
			if (val[0] != 0 && val[0] != 1) {
				// Get next message bit to write.
				String bit = msg.substring(bitsWrit, bitsWrit+1);

				// Change the LSB to the next message bit.
				int mask = 1; // Mask of 1's with 0 in LSB.
				mask = ~mask;
				int newVal = (val[0] & mask) | Integer.decode(bit);
				// Stored encoded coefficient.
				dcts.put(ind,  0, newVal);
				bitsWrit++;
			}
		}
	}

	/**
	 * Decodes DCT coefficients with Outguess. Visits the DCT coefficients in
	 * pseudo-random order determined by pass (the seed). Stores the
	 * least significant bit at visited coefficients to and stops when a null
	 * is read, or the maximum message length has been reached. Returns the
	 * resulting string of characters.
	 * @param DCT coefficients (8-bits signed) of image to encode.
	 * @param Hashed password (int form of password).
	 * @return Decoded message.
	 */
	private static String outguessDecode(Mat dcts, int pass) {
		// Keep track of coefficients written to so we do not overwrite them.
		TreeSet<Integer> visited = new TreeSet<Integer>();
		// Set PRNG's seed
		Random rand2 = new Random();
		rand2.setSeed(pass);

		// Count the maximum number of bits that can be encoded (non-0,
		// non-1 coefficients).
		int maxbits = dcts.rows() - Core.countNonZero(dcts);
		Mat oneVals = new Mat();
		Core.compare(dcts, new Scalar(1), oneVals, Core.CMP_EQ);
		maxbits -= Core.countNonZero(oneVals);

		byte[] msgBytesIn = new byte[maxbits];

		int bytesRead = 0;
		int lastByte = 1;
		int bitNo = 0;
		int curByte = 0;
		while(lastByte != 0 && bytesRead < msgBytesIn.length) {
			// Get next pseudo-random coefficient index.
			int ind = rand2.nextInt(dcts.rows());
			// Make sure we do not read a message bit more than once.
			if (visited.contains(ind)) {
				continue;
			} else {
				visited.add(ind);
			}

			byte[] val = {0};
			dcts.get(ind, 0, val);
			// Only read LSB if coefficient is neither 1 nor 0.
			if (val[0] != 0 && val[0] != 1) {
				// Bit manipulation to extract LSB.
				int mask = 1;
				curByte |= (val[0] & mask) << (7-bitNo);
				bitNo++;

				if (bitNo == 8) { // If full byte read, add it to message.
					msgBytesIn[bytesRead] = (byte)curByte;
					bytesRead++;
					lastByte = curByte;
					curByte = 0;
					bitNo = 0;
				}
			}

		}
		String s = new String(msgBytesIn, StandardCharsets.US_ASCII);
		return s.substring(0, bytesRead); // Do not return full string buffer.
	}

	/**
	 * Hashes the password to an int to be used as a seed for the
	 * pseudo-random number generator.
	 * @param password to be hashed.
	 * @return corresponding int.
	 */
	private static int getSeedFromPass(String pass) {
		MessageDigest md;
		try {
			md = MessageDigest.getInstance("SHA-256");
		} catch (NoSuchAlgorithmException e) {
			System.err.println(e.getMessage());
			return -1;
		}
		md.update(pass.getBytes());
		ByteBuffer buff = ByteBuffer.wrap(md.digest());
		return buff.getInt();
	}

	/**
	 * Encodes provided message into the image located at pathIn using Outguess with
	 * the provided password. Resulting stegoimage created is pathOut (set as a png).
	 * @param path to cover image.
	 * @param path for stegoimage.
	 * @param message to be encoded.
	 * @param password for encoding/decoding message.
	 */
	public static void encodeImage(String pathIn, String pathOut, String msg, String pass) {
		// Read image
		Mat imIn = Highgui.imread(pathIn);

		if (imIn.channels() < 3) {
			return;
		}

		Mat blockDcts = im2dcts(imIn);

		if (!msg.isEmpty()) {
			// Get message's binary representation.
			String msgBin = string2bin(msg);
			// Hash password to get int used as seed for PRNG.
			int seed = getSeedFromPass(pass);

			// Count the maximum number of bits that can be encoded (non-0,
			// non-1 coefficients).
			int maxbits = blockDcts.rows() - Core.countNonZero(blockDcts);
			Mat oneVals = new Mat();
			Core.compare(blockDcts, new Scalar(1), oneVals, Core.CMP_EQ);
			maxbits -= Core.countNonZero(oneVals);
			System.out.println("Max bits: " + maxbits);

			// Encode message in DCT coefficients.
			outguessEncode(blockDcts, msgBin, seed, maxbits);

			// Uncomment to check message extracted from coeffificents.
			// String msgOut = outguessDecode(blockDcts, seed);
			// System.out.println("Decoded: " + msgOut);
		}

		// Get stegoimage
		Mat imOut = dcts2im(blockDcts, imIn.size());

		// Return image to original number format and color space.
		imOut.convertTo(imOut, imIn.type());
		Imgproc.cvtColor(imOut, imOut, Imgproc.COLOR_YCrCb2BGR);

		// Save Image with lossless image format.
		int[] params = {Highgui.CV_IMWRITE_PNG_COMPRESSION, 0};
		Highgui.imwrite(pathOut, imOut, new MatOfInt(params));

		/*
		// Code adapted from opencv forum to output image to file with java.
		// Done to test whether writing to file was the cause for some of
		// wrongly encoded messages. Does not seem to be the cause which is
		// likely type conversions (signed 8-bit ints to 32 bit floats needed
		// to use OpenCV's dct/idct).
		byte[] b = new byte[imOut.channels() * imOut.rows() * imOut.cols()];
		imOut.get(0,0,b);
		BufferedImage image = new BufferedImage(imOut.cols(), imOut.rows(), BufferedImage.TYPE_3BYTE_BGR);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);
		File of = new File("/home/dom/workspace/outguess/trainValid/K.bmp");
		try {
			ImageIO.write(image, "bmp", of);
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		*/
		// Clean memory.
		imIn.release();
		imOut.release();
		blockDcts.release();
		System.gc();
	}

	/**
	 * Encodes a message drawn from msgs of length (sat * the maximum number of characters)
	 * into the image located at pathIn using Outguess with the provided password. Resulting
	 * stegoimage created is pathOut (set as a png). Used in @see generateSaturatedIms.
	 * @param path to cover image.
	 * @param path for stegoimage.
	 * @param wrapped buffer with text for message.
	 * @param password for encoding/decoding message.
	 * @param saturation (expected from 0 to 1). Fraction of available bits to write.
	 */
	public static void encodeImage(String pathIn, String pathOut, BufferedReader msgs, String pass, double sat) {
		// Read image
		Mat imIn = Highgui.imread(pathIn);

		if (imIn.channels() < 3) {
			return;
		}

		Mat blockDcts = im2dcts(imIn);

		// Hash password to get int used as seed for PRNG.
		int seed = getSeedFromPass(pass);

		// Count the maximum number of bits that can be encoded (non-0,
		// non-1 coefficients).
		int maxbits = blockDcts.rows() - Core.countNonZero(blockDcts);
		Mat oneVals = new Mat();
		Core.compare(blockDcts, new Scalar(1), oneVals, Core.CMP_EQ);
		maxbits -= Core.countNonZero(oneVals);
		int bytesReq = (int)((double)maxbits * sat / 8) - 1;

		// Extract message of correct length from BufferedReader msgs.
		String msg = getNextStr(msgs, bytesReq, true) + "\0";
		System.out.println("Max bits: " + maxbits);

		// Get message's binary representation.
		String msgBin = string2bin(msg);

		// Encode message in DCT coefficients.
		outguessEncode(blockDcts, msgBin, seed, maxbits);

		// Get stegoimage
		Mat imOut = dcts2im(blockDcts, imIn.size());

		// Return image to original number format and color space.
		imOut.convertTo(imOut, imIn.type());
		Imgproc.cvtColor(imOut, imOut, Imgproc.COLOR_YCrCb2BGR);

		// Save Image with lossless image format.
		int[] params = {Highgui.CV_IMWRITE_PNG_COMPRESSION, 0};
		Highgui.imwrite(pathOut, imOut, new MatOfInt(params));

		imIn.release();
		imOut.release();
		blockDcts.release();
		System.gc();
	}

	/**
	 * Decodes a message from the image located at pathIn using Outguess with the
	 * provided password. Resulting message is returned.
	 * @param path to image.
	 * @param password for encoding/decoding message.
	 * @return Decoded message.
	 */
	public static String decodeImage(String pathIn, String pass) {
		Mat imIn = Highgui.imread(pathIn);

		if (imIn.channels() < 3) {
			System.err.println("Image must have 3 channels.");
			return "";
		}
		Mat blockDcts = im2dcts(imIn);

		int seed = getSeedFromPass(pass);

		String msgOut = outguessDecode(blockDcts, seed);
		blockDcts.release();
		System.gc();
		return msgOut;
	}

	/**
	 * Returns a string extracted from BufferedReader br. Assumes br has a mark at
	 * the beginning. If we reach the end of the buffer, we resume reading from the
	 * mark. Assumes br has at least maxlen characters.
	 * @param Text from which string is extracted.
	 * @param maximum length of string in characters.
	 * @param If true, length read from br is maxlen, else it's drawn from uniform
	 *        distribution from 1 to maxlen.
	 * @return Extracted string.
	 */
	private static String getNextStr(BufferedReader br, int maxlen, boolean fixed) {
		Random rand = new Random();
		char[] str = new char[maxlen];
		try {
			int len;
			if (fixed) {
				len = maxlen;
			} else {
				len = rand.nextInt(maxlen);
			}
			// Will loop trying to read len. Ensure br has at least maxlen chars.
			while (br.read(str, 0, len) == -1) {
				br.reset();
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return "ErrInPlaceOfStr";
		}
		return String.valueOf(str);
	}

	/**
	 * Uses trained SVM to predict whether image at pathIn is a stegoimage. Predicts
	 * it is a stegoimage if dist <= 0.
	 * @param Trained SVM model.
	 * @param Path to image to analyze.
	 * @return Signed distance to SVM classification boundary.
	 */
	public static double steganalyze(CvSVM mdl, String pathIn) {
		// Model will have to be loaded at start of app.
		Mat feat = extractFeaturesBlock(pathIn);
		return mdl.predict(feat, true);
	}

	/**
	 * Returns confidence in SVM classification. Confidence is the ratio of
	 * in-class samples that had a distance at least as great as dist determined
	 * during training. For example, say a distance >= -0.0001 contained 59% of
	 * stegoimages in training set, then our confidence is 59%. Confidence as
	 * described was well fit in Matlab by a logistic model, so parameterized
	 * confidence is returned below.
	 * @param Signed distance to SVM classification boundary.
	 * @return Confidence (from 0 to 1) of prediction.
	 */
	public static double getSteganalysisConfidence(double dist) {
		double a0, a1, a2;
		if (dist <= 0) { // Is a Stegoimage
			// a = [0.991488706152358	76.2559004631591	-0.0136845517912720]
			a0 = 0.991488706152358;
			a1 = -76.2559004631591;
			a2 = 0.0136845517912720;
			// predicted = @(a,xdata) a(1)./(1 + exp(-a(2) .* (-xdata - a(3))));
		} else {
			// a = [0.997349130691027	30.7335223307588	-0.0633927707102449]
			a0 = 0.997349130691027;
			a1 = 30.7335223307588;
			a2 = -0.0633927707102449;
		}
		return a0 / (1.0 + Math.exp((-a1 *(dist -a2))));
	}

	/**
	 * Encodes images in input rootPath with messages drawn from specified
	 * text file, and passwords also from specified text files. Messages will
	 * be encoded to be a fraction sat of available DCT coefficients for encoding.
	 * Then, an SVM model's prediction results will be saved to a log file.
	 * Used to test the sensitivity in steganalysis of an SVM to how "saturated"
	 * the DCT coefficients are.
	 * @param Desired coefficient saturation of generated images.
	 */
	private static void generateSaturatedIms(double sat) {
		// Prepare resources.
		int maxPassLen = 16;

		// Path to input images.
		String imRootPath = "/home/dom/workspace/outguess/small";
		// Path for output log file and images.
		String outputPath = "/home/dom/workspace/outguess/smallsat3/";
		// Path to text from which to extract passwords.
		String passPath = "/home/dom/workspace/outguess/picounoc_le_maudit_lemay.txt";
		// Path to text from which to extract messages.
		String msgPath = "/home/dom/workspace/outguess/84.txt";

		// Load trained SVM model to test.
		CvSVM mdl = new CvSVM();
		mdl.load("/home/dom/workspace/outguess/svm/svm0_04Fin.dat");

		// Prepare log file for prediction results.
		File logFile = new File(outputPath + "log" + (int)(sat*100) + ".csv");
		BufferedWriter log;
		try {
			log = new BufferedWriter(new FileWriter(logFile));
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}

		File rootDir = new File(imRootPath);
		if (!rootDir.isDirectory()) {
			throw(new Error("Must provide a directory as image directory."));
		}
		File outDir = new File(outputPath);
		if (!outDir.isDirectory()) {
			throw(new Error("Must provide a directory as output directory."));
		}
		File passFile = new File(passPath);
		if (!passFile.exists() && !passFile.isFile()) {
			throw(new Error("Must provide a valid file for passwords."));
		}
		BufferedReader passes;
		try {
			passes = new BufferedReader(new FileReader(passFile));
			passes.mark((int)passFile.length());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}
		File msgFile = new File(msgPath);
		if (!msgFile.exists() && !msgFile.isFile()) {
			throw(new Error("Must provide a valid file for messages."));
		}

		BufferedReader msgs;
		try {
			msgs = new BufferedReader(new FileReader(msgFile));
			msgs.mark((int)msgFile.length());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}

		// Iterate over images in imRootDir
		File[] ims = rootDir.listFiles();

		System.out.println("Starting to generate testing set with saturation " + sat);
		int imCtr = 0;
		int imTot = ims.length;
		for (File f : ims) {
			imCtr++;
			System.out.println("Generating file " + imCtr + " of " + imTot);
			String pathIn = f.getAbsolutePath();
			String pathOut = outputPath;
			String pass = "";

			pathOut += "im_" + (int)(sat*100) + ".png";
			// Get random password of length 16.
			pass = getNextStr(passes, maxPassLen, true);
			try {
				encodeImage(pathIn, pathOut, msgs, pass, sat);

				// Predict whether stegoimage.
				double dist = steganalyze(mdl, pathOut);
				// Store results.
				log.write(imCtr + "," + dist + "," + getSteganalysisConfidence(dist));
				log.newLine();
			} catch (Exception e) {
				System.err.println(e.getMessage());
				pathIn += "$$ERROR$$";
			}
		}
		// Close log file.
		try {
			log.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
	}

	/**
	 * Encodes images in input rootPath with messages drawn from specified
	 * text file, and passwords also from specified text files. Message and
	 * password lengths will be of random lengths (drawn from a uniform
	 * distribution of specified length).
	 * Used to generate a training/test set for training a stegoimage detector.
	 */
	private static void generateTrainSet() {
		// Prepare resources
		int maxPassLen = 16;
		int maxMsgLen = 256;
		// Path to input images.
		String imRootPath = "/home/dom/workspace/outguess/big";
		// Path for output log file and images.
		String outputPath = "/home/dom/workspace/outguess/train2/";
		// Path to text from which to extract passwords.
		String passPath = "/home/dom/workspace/outguess/picounoc_le_maudit_lemay.txt";
		// Path to text from which to extract messages.
		String msgPath = "/home/dom/workspace/outguess/histoire_de_la_rev_fr_thiers.txt";

		// Create log file to match input and output images, track passwords, messages
		// and file names.
		File logFile = new File(outputPath + "log.csv");
		BufferedWriter log;
		try {
			log = new BufferedWriter(new FileWriter(logFile));
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}

		File rootDir = new File(imRootPath);
		if (!rootDir.isDirectory()) {
			throw(new Error("Must provide a directory as image directory."));
		}
		File outDir = new File(outputPath);
		if (!outDir.isDirectory()) {
			throw(new Error("Must provide a directory as output directory."));
		}
		File passFile = new File(passPath);
		if (!passFile.exists() && !passFile.isFile()) {
			throw(new Error("Must provide a valid file for passwords."));
		}
		BufferedReader passes;
		try {
			passes = new BufferedReader(new FileReader(passFile));
			passes.mark((int)passFile.length());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}
		File msgFile = new File(msgPath);
		if (!msgFile.exists() && !msgFile.isFile()) {
			throw(new Error("Must provide a valid file for messages."));
		}

		BufferedReader msgs;
		try {
			msgs = new BufferedReader(new FileReader(msgFile));
			msgs.mark((int)msgFile.length());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}

		int withNo = 0; // Count stegoimages
		int withoutNo = 0; // Count non-stegoimages
		// Iterate over images in imRootDir
		File[] ims = rootDir.listFiles();

		System.out.println("Starting to generate training set.");
		int imCtr = 0;
		int imTot = ims.length;
		Random setRand = new Random();
		for (File f : ims) {
			imCtr++;
			System.out.println("Generating file " +
					Integer.toString(imCtr) + " of " + Integer.toString(imTot));

			String pathIn = f.getAbsolutePath();

			String pathOut = outputPath;
			String pass = "";
			String msg = "";

			// Flip a coin to determine whether to generate stegoimage or not.
			if (setRand.nextBoolean()) {
				withNo++;
				pathOut += "_y_" + Integer.toString(withNo) + ".png";
				// Get message and password randomly
				pass = getNextStr(passes, maxPassLen, false);
				msg = getNextStr(msgs, maxMsgLen, false) + "\0";
			} else {
				withoutNo++;
				pathOut += "_n_" + Integer.toString(withoutNo) + ".png";
			}
			try {
				encodeImage(pathIn, pathOut, msg, pass);
				if (pathOut.contains("_y_")) { // Ensure encoding went well.
					String msgOut = decodeImage(pathOut, pass);
					if (msgOut != msg) {
						System.out.println("msg in: " + msg);
						System.out.println("msg out: " + msgOut);
					}
				}
			} catch (Exception e) {
				System.err.println(e.getMessage());
				pathIn += "$$ERROR$$";
			}
			// Log details of generated training image.
			try {
				log.write(pathOut + ", " + pathIn + ", \"" + pass + "\", \"" + msg + "\"");
				log.newLine();
			} catch (IOException e) {
				System.err.println(e.getMessage());
				return;
			}
		}
		// Close log file.
		try {
			log.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
	}

	/**
	 * Extracts features for all images in dataPath (expects .png), and saves
	 * stegoimage status, as well as features to file. Generated files intended
	 * to be used to train a model to detect stegoimages. Uses extractFeaturesBlock
	 * in final version.
	 * @param Path to input images.
	 * @param Path where features and true classification should be saved.
	 *        eg: "$PATH/feat03.dat" (expects "feat" in filename.
	 */
	private static void extractTrainFeatures(String dataPath, String outPath) {
		int featSz = 6; // Specify size of feature vector.

		File rootDir = new File(dataPath);
		if (!rootDir.isDirectory()) {
			throw(new Error("Must provide a directory as image directory."));
		}
		File[] ims = rootDir.listFiles();

		System.out.println("Starting to extracting features from training set.");
		int imCtr = 0;
		int imTot = ims.length;
		Mat feats = Mat.zeros(ims.length, featSz, CvType.CV_32FC1);
		Mat responses = Mat.zeros(ims.length, 1, CvType.CV_32SC1);
		int[] stego = {1};
		int[] notStego = {0};
		// Iterate over images and add their features to feature matrix.
		for (File f : ims) {
			if (!f.getName().contains(".png")) {
				continue;
			}
			imCtr++;
			System.out.println("Im " + Integer.toString(imCtr) + " of " + Integer.toString(imTot));
			Mat curRow = feats.submat(imCtr-1, imCtr, 0, featSz);

			// Final version uses extractFeaturesBlock for feature extraction.
			Mat curFeat = extractFeaturesBlock(f.getAbsolutePath());

			curFeat.copyTo(curRow);
			// Store true classification from file name.
			if (f.getName().contains("_y_")) {
				responses.put(imCtr-1, 0, stego);
			} else {
				responses.put(imCtr-1, 0, notStego);
			}
		}
		// Save features and true classifications to files.
		try {
			// Save features to file.
			FileOutputStream fos = new FileOutputStream(outPath);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			ArrayList<Float> toList = new ArrayList<Float>();
			// Print first row to ensure upon reloading that data was
			// saved correctly.
			System.out.println(feats.submat(0,1,0,featSz).dump());
			// Data must be a single column for conversion to vector.
			feats = feats.t();
			feats = feats.reshape(1, 1);
			Converters.Mat_to_vector_float(feats.t(), toList);
			oos.writeObject(toList);
			oos.close();

			// Save true classifications to file.
			String respPath = outPath.replace("feat", "resp");
			fos = new FileOutputStream(respPath);
			oos = new ObjectOutputStream(fos);
			ArrayList<Integer> respToList = new ArrayList<Integer>();
			// Print first 10 predictions for verification upon reloading.
			System.out.println(responses.submat(0,10,0,1).dump());
			Converters.Mat_to_vector_int(responses, respToList);
			oos.writeObject(respToList);
			oos.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		System.out.println("Done extracting features from " + dataPath);
	}

	/**
	 * First version of feature extraction based on general image statistics.
	 * Extracts image moments for horizontal, vertical and diagonal features.
	 * Repeats this for four different scales.
	 * Obtains 9 moments for a set direction, scale, and channel.
	 *
	 * @param Path to input image.
	 * @return Feature vector of length 324.
	 */
	private static Mat extractFeatures(String imPath) {
		Mat feats = Mat.zeros(1, 324, CvType.CV_32FC1);

		// Read image as YCrCb
		Mat im = Highgui.imread(imPath);
		Imgproc.cvtColor(im, im, Imgproc.COLOR_BGR2YCrCb);
		Mat ch = new Mat();
		for (int chNo = 0; chNo < 3; chNo++) { // For each channel
			Core.extractChannel(im, ch, chNo);
			// 4 scales
			double[] sigmaVec = {0.0, 3.0, 15.0, 29.0};
			int scaleCtr = 0;
			Mat chHz = new Mat(); // Horizontal features for channel.
			Mat chVert = new Mat(); // Vertical features for channel.
			Mat chDiag = new Mat(); // Diagonal features for channel.
			for (int dCtr = 0; dCtr < sigmaVec.length; dCtr++) { // For each scale
				scaleCtr++;
				int baseIdx = chNo * 108 + (scaleCtr-1) * 27;
				Core.extractChannel(im, ch, 0);
				// Select scale through Gaussian Blurring which will be followed by
				// Edge detection (Scharr operator).
				if (dCtr != 0) {
					double sigmaX = sigmaVec[dCtr];
					Imgproc.GaussianBlur(ch, ch, new Size(sigmaX / 3, sigmaX / 3), sigmaX);
				}
				// 3 directions (hz, vert, diag)
				MatOfDouble mean = new MatOfDouble();
				MatOfDouble stddev = new MatOfDouble();
				Moments m = new Moments();

				// Horizontal stats
				Imgproc.Scharr(ch, chHz, ch.depth(), 1, 0);

				Core.meanStdDev(chHz, mean, stddev);
				m = Imgproc.moments(chHz);

				feats.put(0, baseIdx+0, mean.get(0, 0));
				feats.put(0, baseIdx+1, stddev.get(0, 0));
				feats.put(0, baseIdx+2, m.get_m00());
				feats.put(0, baseIdx+3, m.get_mu02());
				feats.put(0, baseIdx+4, m.get_mu03());
				feats.put(0, baseIdx+5, m.get_mu11());
				feats.put(0, baseIdx+6, m.get_mu12());
				feats.put(0, baseIdx+7, m.get_mu20());
				feats.put(0, baseIdx+8, m.get_mu21());

				// Vertical stats
				Imgproc.Scharr(ch, chVert, ch.depth(), 0, 1);

				Core.meanStdDev(chVert, mean, stddev);
				m = Imgproc.moments(chVert);

				feats.put(0, baseIdx+9, mean.get(0, 0));
				feats.put(0, baseIdx+10, stddev.get(0, 0));
				feats.put(0, baseIdx+11, m.get_m00());
				feats.put(0, baseIdx+12, m.get_mu02());
				feats.put(0, baseIdx+13, m.get_mu03());
				feats.put(0, baseIdx+14, m.get_mu11());
				feats.put(0, baseIdx+15, m.get_mu12());
				feats.put(0, baseIdx+16, m.get_mu20());
				feats.put(0, baseIdx+17, m.get_mu21());

				// Diagonal stats
				Imgproc.Scharr(ch, chDiag, ch.depth(), 0, 1);

				Core.meanStdDev(chDiag, mean, stddev);
				m = Imgproc.moments(chDiag);

				feats.put(0, baseIdx+18, mean.get(0, 0));
				feats.put(0, baseIdx+19, stddev.get(0, 0));
				feats.put(0, baseIdx+20, m.get_m00());
				feats.put(0, baseIdx+21, m.get_mu02());
				feats.put(0, baseIdx+22, m.get_mu03());
				feats.put(0, baseIdx+23, m.get_mu11());
				feats.put(0, baseIdx+24, m.get_mu12());
				feats.put(0, baseIdx+25, m.get_mu20());
				feats.put(0, baseIdx+26, m.get_mu21());
			}
			chHz.release();
			chVert.release();
			chDiag.release();
		}

		im.release();
		ch.release();

		System.gc();
		return feats;
	}
	// In final app

	/**
	 * Final version of feature extraction based on Blockiness of two
	 * images. Blockiness is the sum of the intensity difference between
	 * neighboring pixels over the boundary of a grid of 8x8 pixel blocks.
	 * Blockiness is extracted from image of interest, and as a proxy
	 * for the unencoded image, we lightly compress the image, then shift
	 * it by 4 pixels horizontally and vertically (assuming here that
	 * blockiness at point (x1, y1) should not be much different from
	 * blockiness at point (x1 + 4, y1 + 4).
	 *
	 * Returns a vector of length 6 with the blockiness for each channel of
	 * of the two images described in the above paragraph.
	 *
	 * @param Path to input image.
	 * @return Feature vector of length 6.
	 */
	private static Mat extractFeaturesBlock(String imPath) {
		Mat feats = Mat.zeros(1, 6, CvType.CV_32FC1);
		Mat im = Highgui.imread(imPath);
		// Image that will proxy for clean image.
		Mat imBgr = new Mat();
		im.copyTo(imBgr);

		// Extract blockiness of original image.
		Imgproc.cvtColor(im, im, Imgproc.COLOR_BGR2YCrCb);
		Mat ch = new Mat();
		int M = (int) Math.floor((im.rows() - 1) / 8);
		int N = (int) Math.floor((im.cols() - 1) / 8);
		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(im, ch, chNo);
			double blo = getBlockiness(ch, M, N); // blockiness
			feats.put(0, chNo, blo / N / M);
		}

		// Compress proxy image.
		MatOfByte buf = new MatOfByte();
		Highgui.imencode(".jpg", imBgr, buf);
		imBgr = Highgui.imdecode(buf, Highgui.CV_LOAD_IMAGE_COLOR);
		Imgproc.cvtColor(imBgr, imBgr, Imgproc.COLOR_BGR2YCrCb);
		// Crop proxy image by 4 pixels horizontally and vertically.
		imBgr = im.submat(4, im.rows(), 4, im.cols());
		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(imBgr, ch, chNo);
			double blo = getBlockiness(ch, M, N); // blockiness
			feats.put(0, 3+chNo, blo / N / M);
		}

		im.release();
		ch.release();
		System.gc();

		return feats;
	}
	// In final app

	/**
	 * Computes and returns blockiness of grayscale image.
	 *
	 * Blockiness is the sum of the intensity difference between
	 * neighboring pixels over the boundary of a grid of 8x8 pixel blocks.
	 *
	 * @param Input grayscale matrix.
	 * @param Number of 8x8 blocks horizontally.
	 * @param Number of 8x8 blocks vertically.
	 * @return Feature vector of length 6.
	 */
	private static double getBlockiness(Mat ch, int M, int N) {
		double blo = 0.0; // blockiness
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < ch.cols(); j++) {
				double[] prev = ch.get(8*i,j);
				double[] next = ch.get(8*i+1,j);;
				blo += Math.abs(prev[0] - next[0]);
			}
		}
		for (int i = 0; i < ch.rows(); i++) {
			for (int j = 0; j < N; j++) {
				double[] prev = ch.get(i,8*j);
				double[] next = ch.get(i,8*j+1);
				blo += Math.abs(prev[0] - next[0]);
			}
		}
		return blo;
	}


	/**
	 * Version of feature extraction based on Blockiness of five
	 * images. Blockiness is the sum of the intensity difference between
	 * neighboring pixels over the boundary of a grid of 8x8 pixel blocks.
	 * Follows the procedure described by Fidrich et al. (2000).
	 *
	 * Blockiness is extracted from image of interest (I0), I0 with a
	 * message of maximal length encoded (I1), a proxy
	 * for the unencoded image (I2) (lightly compress the image, then shift
	 * it by 4 pixels horizontally and vertically), I2 with a
	 * message of maximal length encoded once (I3), I3 with a
	 * message of maximal length encoded (I4).
	 *
	 * Returns a vector of length 15 with the blockiness for each channel of
	 * of the five images described in the above paragraph.
	 *
	 * @param Path to input image.
	 * @return Feature vector of length 15.
	 */
	private static Mat extractFeaturesFullBlock(String imPath) {
		Mat feats = Mat.zeros(1, 12, CvType.CV_32FC1);
		Mat im = Highgui.imread(imPath);
		Mat imBgr = new Mat();
		im.copyTo(imBgr);
		Imgproc.cvtColor(im, im, Imgproc.COLOR_BGR2YCrCb);
		Mat ch = new Mat();
		int M = (int) Math.floor((im.rows() - 1) / 8);
		int N = (int) Math.floor((im.cols() - 1) / 8);

		// Get blockiness of original image.
		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(im, ch, chNo);
			double blo = getBlockiness(ch, M, N);
			feats.put(0, chNo, blo);
		}

		// Get blockiness of original image with message of maximal length.
		Mat blockDcts = im2dcts(im);
		String msgBin = string2bin("Neverending");
		int maxbits = blockDcts.rows() - Core.countNonZero(blockDcts);
		Mat oneVals = new Mat();
		Core.compare(blockDcts, new Scalar(1), oneVals, Core.CMP_EQ);
		maxbits += Core.countNonZero(oneVals);
		while (msgBin.length() < maxbits) {
			msgBin += msgBin;
		}
		outguessEncode(blockDcts, msgBin, 12, maxbits);
		Mat imOut = dcts2im(blockDcts, im.size());
		imOut.convertTo(imOut, im.type());

		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(imOut, ch, chNo);
			double blo = getBlockiness(ch, M, N); // blockiness
			feats.put(0, 3+chNo, blo);
		}

		// Generate image compressed and cropped by 4 pixels.
		MatOfByte buf = new MatOfByte();
		Highgui.imencode(".jpg", imBgr, buf);
		imBgr = Highgui.imdecode(buf, Highgui.CV_LOAD_IMAGE_COLOR);
		Imgproc.cvtColor(imBgr, imBgr, Imgproc.COLOR_BGR2YCrCb);

		// Get blockiness of compressed and cropped image.
		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(imBgr, ch, chNo);
			double blo = getBlockiness(ch, M, N); // blockiness
			feats.put(0, 6+chNo, blo);
		}

		// Encode maximal length message in compressed, cropped image.
		blockDcts = im2dcts(imBgr);
		msgBin = string2bin("Neverending");
		maxbits = blockDcts.rows() - Core.countNonZero(blockDcts);
		oneVals = new Mat();
		Core.compare(blockDcts, new Scalar(1), oneVals, Core.CMP_EQ);
		maxbits += Core.countNonZero(oneVals);
		while (msgBin.length() < maxbits) {
			msgBin += msgBin;
		}
		outguessEncode(blockDcts, msgBin, 12, maxbits);
		imOut = dcts2im(blockDcts, imBgr.size());
		imOut.convertTo(imOut, imBgr.type());

		// Get blockiness of compressed, cropped image with maximal
		// length message.
		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(imOut, ch, chNo);
			double blo = getBlockiness(ch, M, N); // blockiness
			feats.put(0, 9+chNo, blo);
		}

		// Get blockiness of compressed, cropped image with maximal
		// length messages encoded twice.

		// Encode different maximal length message.
		blockDcts = im2dcts(imOut);
		msgBin = string2bin("[[[Clamoring]]]");
		maxbits = blockDcts.rows() - Core.countNonZero(blockDcts);
		oneVals = new Mat();
		Core.compare(blockDcts, new Scalar(1), oneVals, Core.CMP_EQ);
		maxbits += Core.countNonZero(oneVals);
		while (msgBin.length() < maxbits) {
			msgBin += msgBin;
		}
		outguessEncode(blockDcts, msgBin, 12, maxbits);

		imOut = dcts2im(blockDcts, imBgr.size());
		imOut.convertTo(imOut, imBgr.type());
		for (int chNo = 0; chNo < 3; chNo++) {
			Core.extractChannel(imOut, ch, chNo);
			double blo = getBlockiness(ch, M, N); // blockiness
			feats.put(0, 12+chNo, blo);
		}


		im.release();
		imOut.release();
		imBgr.release();
		blockDcts.release();
		ch.release();

		System.gc();
		return feats;
	}


	/**
	 * Runs a Bootstrap on a classifier of type mdlType with training data feat,
	 * classification resp. Saves data relating to every iteration of the Bootstrap.
	 * @param Training data observations.
	 * @param Training data classification.
	 * @param Path to which to save trained classifiers and related data.
	 * @param Model type ('svm', 'boost', 'bayes').
	 */
	public static void trainModel(Mat feat, Mat resp, String savePath, String mdlType) {
		System.out.println("Starting to train SVM");

		// Initialize models.
		CvSVM svmMdl = new CvSVM();
		CvBoost boostMdl = new CvBoost();
		CvNormalBayesClassifier bayesMdl = new CvNormalBayesClassifier();
		CvStatModel mdl = svmMdl;
		CvBoostParams bParams = new CvBoostParams();
		CvSVMParams sParams = new CvSVMParams();
		if (mdlType == "boost") {
			mdl = boostMdl;
			bParams.set_cv_folds(10);
			// Set SVM parameters if desired. Default is Radial Basis Function Kernel.
		/*else if (mdlType == "svm") {
			//sParams.set_svm_type(CvSVM.C_SVC);
			//sParams.set_kernel_type(CvSVM.LINEAR);
			//sParams.set_kernel_type(CvSVM.POLY);
			//sParams.set_degree(2.0);
			//sParams.set_kernel_type(CvSVM.SIGMOID);*/
		} else if (mdlType == "boost") {
			mdl = boostMdl;
			bParams.set_cv_folds(10);
		} else if (mdlType == "bayes") {
			mdl = bayesMdl;
		}
		Mat sampleIdx = new Mat(feat.rows(), 1, CvType.CV_8SC1);
		Mat varIdx = Mat.ones(feat.cols(), 1, CvType.CV_8SC1);

		for (int iter = 0; iter < 1; iter++) {
			// Select random sample with replacement.
			Core.randu(sampleIdx, 0, 2);
			try {
				// Train Model
				if (mdlType == "svm") {
					svmMdl.train_auto(feat, resp, varIdx, sampleIdx, sParams);
				} else if (mdlType == "boost") {
					boostMdl.train(feat, 1, resp);
					//boostMdl.train(feat, 0, resp, varIdx, sampleIdx, new Mat(), new Mat(), bParams, true);
				} else if (mdlType == "bayes") {
					bayesMdl.train(feat, resp, varIdx, sampleIdx, false);
				}
				System.out.println("Done training iteration " + iter);

				// Save model
				mdl.save(savePath.replace("0", Integer.toString(iter)));

				// Save which samples of the training set were used.
				FileOutputStream fos = new FileOutputStream(savePath.replace("svm0", "idx"+Integer.toString(iter)));
				ObjectOutputStream oos = new ObjectOutputStream(fos);
				ArrayList<Integer> idxToList = new ArrayList<Integer>();
				Mat sampleIdxSave = new Mat();
				sampleIdx.convertTo(sampleIdxSave, CvType.CV_32SC1);
				Converters.Mat_to_vector_int(sampleIdxSave, idxToList);
				oos.writeObject(idxToList);
				oos.close();

				System.out.println("Done saving iteration " + iter);

				// Gauge classification performance for iteration.
				Mat pred = Mat.zeros(resp.size(), CvType.CV_32FC1);

				if (mdlType == "svm") {
					svmMdl.predict_all(feat, pred);
				} else if (mdlType == "boost") {
					for (int sampleNo = 0; sampleNo < feat.rows(); sampleNo++) {
						double guess = boostMdl.predict(feat.submat(sampleNo, sampleNo+1,0,feat.cols()));
						pred.put(sampleNo, 0, guess);
					}
				} else if (mdlType == "bayes") {
					bayesMdl.predict(feat, pred);
				}
				Mat eval = new Mat();

				// Output Accuracy (true positive + true negative), and false negative rate
				resp.convertTo(resp, CvType.CV_32FC1);
				Core.compare(pred, resp, eval, Core.CMP_EQ);
				System.out.print("Accuracy: ");
				System.out.println((double)Core.countNonZero(eval) / (double)eval.rows());

				Core.compare(pred, resp, eval, Core.CMP_LT);
				System.out.print("False negative rate: "); // 0 when was 1
				System.out.println((double)Core.countNonZero(eval) / (double)eval.rows());

			/*	Uncomment following to save details of prediction to csv.
			 * File respFile = new File("./pred04_final_" + iter + ".csv");
				PrintWriter log;
				try {
					log = new PrintWriter(new BufferedWriter(new FileWriter(respFile)));
					for (int sampleNo = 0; sampleNo < feat.rows(); sampleNo++) {
						double guess = svmMdl.predict(feat.submat(sampleNo, sampleNo+1,0,feat.cols()), true);
						log.print(guess);
						log.print(",");
					}
				} catch (Exception e) {
					System.err.println(e.getMessage());
					return;
				}
				log.close();*/
			} catch (Exception e) {
				System.err.println(e.getMessage());
				continue;
			}
		}
		System.out.println("Done training Model");
	}

	/**
	 * Saves trained classifier predictions and vector of binary classifications
	 * to CSVs for processing elsewhere.
	 * @param Path to classifier.
	 * @param Training data features.
	 * @param Training data classifications.
	 */
	private static void resToCsv(String mdlPath, Mat feat, Mat resp) {
		// Save responses to CSV.
		File respFile = new File("./resp04.csv");
		PrintWriter log;
		try {
			log = new PrintWriter(new BufferedWriter(new FileWriter(respFile)));
			System.out.println(resp);
			for (int i = 0; i < feat.rows(); i++) {
				log.print(resp.get(i, 0)[0]);
				log.print(",");
			}

		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}
		log.close();

		// Save predictions to CSV.
		// Load model.
		CvSVM svmMdl = new CvSVM();
		svmMdl.load(mdlPath);

		System.out.println("num vars used: " + svmMdl.get_var_count());
		respFile = new File("./pred04.csv");
		try {
			log = new PrintWriter(new BufferedWriter(new FileWriter(respFile)));
			// Predict and write.
			for (int sampleNo = 0; sampleNo < feat.rows(); sampleNo++) {
				double guess = svmMdl.predict(feat.submat(sampleNo, sampleNo+1,0,feat.cols()), true);
				log.print(guess);
				log.print(",");
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
			return;
		}
		log.close();
	}


	public static void main(String[] args) {
		// Load opencv library in Linux.
		System.load(System.getProperty("user.dir") + "/libopencv_java2413.so");
		// Load opencv library in Windows.
		//System.load(System.getProperty("user.dir") +"\\opencv_java2413.dll");

		// Encode and decode an image.
		// Define message
//		String msg = "End of Quarter!!\0";
//		String pathIn = "./rnc.jpg";
//		String pathOut = "./flashy.png";
//		String pass = "podium!";
//		encodeImage(pathIn, pathOut, msg, pass);
//		System.out.println("Decoded: " + decodeImage("./flashy.png", pass));

		// Uncomment to generate training set.
		//generateTrainSet();

		// Uncomment to test classifier performance versus coefficient saturation.
		//generateSaturatedIms(0.03);

		// Uncomment to extract features from training set.
		//extractTrainFeatures("/home/dom/workspace/outguess/train", "/home/dom/workspace/outguess/svm/feats04.dat");

		// Reload features and classifiations from training set.
		Mat res = new Mat();
		Mat resp = new Mat();
		try {
			// Indicate feature vector size.
			int featSz = 6;

			// Reload features.
			FileInputStream fis = new FileInputStream("./feats04.dat");
			ObjectInputStream ois = new ObjectInputStream(fis);
			ArrayList<Float> lss = (ArrayList<Float>) ois.readObject();
			ois.close();
			res = Converters.vector_float_to_Mat(lss);

			res = res.t();
			res = res.t().reshape(1, featSz).t();

			// Reload responses.
			fis = new FileInputStream("./resps04.dat");
			ois = new ObjectInputStream(fis);
			ArrayList<Integer> lis = (ArrayList<Integer>) ois.readObject();
			ois.close();

			resp = Converters.vector_int_to_Mat(lis);
			// Show first ten classifications for verification.
			System.out.println(resp.submat(0,10,0,1).dump());
		} catch (Exception e) {
			System.out.println(resp);
			System.err.println(e.getMessage());
		}

		// Uncomment to train a model on loaded training set.
		//trainSVM(res, resp, "/home/dom/workspace/outguess/svm/svm0.dat", "svm");

		// Loads a model and classifies input image.
		CvSVM mdl = new CvSVM();
		mdl.load("/home/dom/workspace/outguess/svmFinal.dat");
		double dist = steganalyze(mdl, "/home/dom/workspace/outguess/_y_3.png");
		System.out.println("For _y_3.png, dist is " + dist + " with conf " +
				getSteganalysisConfidence(dist));

	}

}
