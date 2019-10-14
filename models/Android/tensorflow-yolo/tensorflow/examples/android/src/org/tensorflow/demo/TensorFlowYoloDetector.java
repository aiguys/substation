/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Trace;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.env.SplitTimer;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class TensorFlowYoloDetector implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 5;
  private static final float Iou_thread = 0.35f;

  //private static final int NUM_CLASSES = 20;

  //private static final int NUM_BOXES_PER_BLOCK = 5;

  // TODO(andrewharp): allow loading anchors and classes
  // from files.
  /*private static final double[] ANCHORS = {
    1.08, 1.19,
    3.42, 4.41,
    6.63, 11.38,
    9.42, 9.50,
    10.02, 11.31,
    16.62, 16.52
  };*/
  // 一组anchor pair第一个是anchor的宽w，第二个是anchor的高h 例w=10，h=14
  private static final double[] ANCHORS = {
          10, 14,
          23, 27,
          37, 58,
          81, 82,
          135, 169,
          344, 319
  };
    private static final int NUM_BOXES_PER_BLOCK = ANCHORS.length / 2;
/*
  private static final String[] LABELS = {
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
  };
*/
private static final String[] LABELS = {
        "person",
        "hat"
};

  private static final int NUM_CLASSES = LABELS.length;
  // Config values.
  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intValues;
  private float[] floatValues;
  private String[] outputNames;

  private int PF1_blockSize;
  private int PF2_blockSize;

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  /** Initializes a native TensorFlow session for classifying images. */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final int inputSize,
      final String inputName,
      final String outputName,
      final int PF1_blockSize,
      final int PF2_blockSize) {
    TensorFlowYoloDetector d = new TensorFlowYoloDetector();
    d.inputName = inputName; //inputName
    d.inputSize = inputSize;

    // Pre-allocate buffers.
    d.outputNames = outputName.split(",");
    d.intValues = new int[inputSize * inputSize]; // 416 * 416 = 173,056
    d.floatValues = new float[inputSize * inputSize * 3];
    d.PF1_blockSize = PF1_blockSize;
    d.PF2_blockSize = PF2_blockSize;

    d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    return d;
  }

  private TensorFlowYoloDetector() {}

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }


  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    final SplitTimer timer = new SplitTimer("recognizeImage");

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    for (int i = 0; i < intValues.length; ++i) {
      floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
      floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
      floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
    Trace.endSection();

    timer.endSplit("ready for inference");

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    timer.endSplit("ran inference");

    // Copy the first reception field output Tensor back into the output array.
    Trace.beginSection("fetch");
    final int PF1_gridWidth = bitmap.getWidth() / PF1_blockSize; // 416 / 32 = 13
    final int PF1_gridHeight = bitmap.getHeight() / PF1_blockSize;
    // output是13*13感受野尺度上的三维tensor，13*13*(NUM_CLASSES + 5)*3拉成一维array，size=3549
    // gridWidth=gridHeight=13，NUM_BOXES_PER_BLOCK=6代表anchor box pairs，5代表4个位置坐标 + 1 confidence
    final float[] output =
        new float[PF1_gridWidth * PF1_gridHeight * (NUM_CLASSES + 5) * (NUM_BOXES_PER_BLOCK / 2)]; //在13*13尺度上只使用了3组anchors，即6/2=3
    inferenceInterface.fetch(outputNames[0], output); //取出名为outputNames[0]的tensor，读取结果进output字节大小的Flatbuffer； Creates a new float buffer by wrapping the given float array.
    Trace.endSection();

    // Find the best detections.
    final PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            1,
            new Comparator<Recognition>() {
              @Override
              public int compare(final Recognition lhs, final Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    // 开始在13*13的feature map上对每个yolo cell上进行判断是否有物体, offset可以理解为output array的index
    // offset 0:1是x，y坐标，2:3是w，h，4是confidence，5以后（5:6）是class的预测概率
    for (int y = 0; y < PF1_gridHeight; ++y) {
      for (int x = 0; x < PF1_gridWidth; ++x) {
        for (int b = 3; b < NUM_BOXES_PER_BLOCK; ++b) { // b 用于选择anchors，13*13大感受野选择大anchors
          final int offset =
              (PF1_gridWidth * (NUM_BOXES_PER_BLOCK / 2 * (NUM_CLASSES + 5))) * y //13 * 3 * 7 * 13
                  + (NUM_BOXES_PER_BLOCK / 2 * (NUM_CLASSES + 5)) * x // 3 * 7 * 13
                  + (NUM_CLASSES + 5) * (b - 3); //7 * 3

          final float xPos = (x + expit(output[offset + 0])) * PF1_blockSize; //放缩回416的image尺寸
          final float yPos = (y + expit(output[offset + 1])) * PF1_blockSize;
          //根据若干组anchors计算当前x，y坐标的W和H（即以anchor box为前提计算出当前坐标的bounding box）
          final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) ;//保留13*13 feature map的bbox尺寸
          final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) ;

          final RectF rect =
              new RectF(
                  Math.max(0, xPos - w / 2),
                  Math.max(0, yPos - h / 2),
                  Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                  Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(output[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > 0.01 && confidenceInClass != 0.25) { // 原因不明，无时无刻对任何输入都有一个confidence=0.25的person输出调试后应该是output fetch出来的信息有些问题。line226，233，240,244
            LOGGER.i(
                "%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect);
              pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect)); //为priority queue写进output结果
          }
        }
      }
    }

    timer.endSplit("decoded results");
    final List<Recognition> recognitions = new ArrayList<Recognition>();
    /*
    final List<List<Recognition>> recognitions = new ArrayList<List<Recognition>>();
    final ArrayList<Recognition> recognitionsTEMP = new ArrayList<Recognition>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      recognitionsTEMP.add(pq.poll());
    }
    recognitions.add(recognitionsTEMP);
    Trace.endSection(); // "recognizeImage"

    timer.endSplit("processed results"); */

    // Copy the 2-nd reception field output Tensor back into the output array.
    Trace.beginSection("fetch");
    final int PF2_gridWidth = bitmap.getWidth() / PF2_blockSize;
    final int PF2_gridHeight = bitmap.getHeight() / PF2_blockSize;
    // output是26*26感受野尺度上的tensor，gridWidth=gridHeight=26，NUM_BOXES_PER_BLOCK=6代表anchor box pairs，5代表4个位置坐标 + 1 confidence
    final float[] PF2_output =
            new float[PF2_gridWidth * PF2_gridHeight * (NUM_CLASSES + 5) * (NUM_BOXES_PER_BLOCK / 2)]; //在26*26尺度上只使用了3组anchors，即6/2=3
    inferenceInterface.fetch(outputNames[1], PF2_output); //取出名为outputNames[1]的tensor，读取结果进output字节大小的Flatbuffer； Creates a new float buffer by wrapping the given float array.
    Trace.endSection();

    // Find the best detections.
   /* final PriorityQueue<Recognition> pf2_pq =
            new PriorityQueue<Recognition>(
                    1,
                    new Comparator<Recognition>() {
                      @Override
                      public int compare(final Recognition lhs, final Recognition rhs) {
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                      }
                    }); */
    // 开始在26*126的feature map上进行判断是否有物体，offset理解为output array的index
    // offset 0:1是x，y坐标，2:3是w，h，4是confidence，5以后（5:6）是class的预测概率
    for (int y = 0; y < PF2_gridHeight; ++y) {
      for (int x = 0; x < PF2_gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK / 2; ++b) { // b 用于选择anchors，26*26小感受野选择小anchors
          final int offset =
                  (PF2_gridWidth * (NUM_BOXES_PER_BLOCK / 2  * (NUM_CLASSES + 5))) * y //13 * 3 * 7 * 13
                          + (NUM_BOXES_PER_BLOCK / 2  * (NUM_CLASSES + 5)) * x //3 * 7 * 13
                          + (NUM_CLASSES + 5) * b; // 7 * 3

          final float xPos = (x + expit(PF2_output[offset + 0])) * PF2_blockSize;//放缩回416的image尺寸
          final float yPos = (y + expit(PF2_output[offset + 1])) * PF2_blockSize;
          //根据若干组anchors计算当前x，y坐标的W和H（即以anchor box为前提计算出当前坐标的bounding box）
          final float w = (float) (Math.exp(PF2_output[offset + 2]) * ANCHORS[2 * b + 0]) ;//保留26*26 feature map的bbox尺寸
          final float h = (float) (Math.exp(PF2_output[offset + 3]) * ANCHORS[2 * b + 1]) ;

          final RectF rect =
                  new RectF(
                          Math.max(0, xPos - w / 2),
                          Math.max(0, yPos - h / 2),
                          Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                          Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(PF2_output[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = PF2_output[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > 0.01 && confidenceInClass != 0.25) {
            LOGGER.i(
                    "%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect);
            pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect)); //为priority queue写进PF2_output结果
          }
        }
      }
    }

    timer.endSplit("decoded results");

    //过滤两个PF层的输出
    List<Recognition> pq_results = new ArrayList<Recognition>();
    int size_re = pq.size();
    for(int i = 0; i < Math.min(size_re, MAX_RESULTS * 2); i++){
        pq_results.add(pq.poll());
    }
    float Iou;
    for(int i = 0; i < pq_results.size(); i++){
        for(int j = i + 1; j < pq_results.size(); j++){
            if(pq_results.get(i).getTitle() == pq_results.get(j).getTitle()){
                Iou = Iou_Calculation(pq_results.get(i).getLocation(),pq_results.get(j).getLocation());
                if(Iou < Iou_thread)
                    //虽然title相同但IOU低于thread，不作处理
                    continue;
                else
                    if(pq_results.get(i).getConfidence() > pq_results.get(j).getConfidence()){
                        //相同title抛弃低confidence的那一项j
                        pq_results.remove(j);
                        j--;} //将j的值退一位，从list中相同位置继续遍历
                    else {
                        //相同title抛弃低confidence的那一项i
                        pq_results.remove(i);
                        i--;
                        j = 0;} //重置j，重新开始遍历
            }
            else
                continue;
        }
    }
   for (int i = 0; i < Math.min(pq_results.size(), MAX_RESULTS); i++) {
       recognitions.add(pq_results.get(i));
   }
              //final ArrayList<Recognition> recognitionsTEMP2 = new ArrayList<Recognition>();
   // for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
   //     recognitions.add(pq.poll());
   // }
   // recognitions.add(recognitionsTEMP2);

    Trace.endSection(); // "recognizeImage"

    timer.endSplit("processed results");

    return recognitions;
  }

  private static float Iou_Calculation(RectF s1, RectF s2){
      float iou;
      if(!s1.intersect(s2))
          return 0;

      float union;
      if(s1.left > s2.left) {
          if (s1.top > s2.top) {
              union = s1.height() * s1.width() + s2.height() * s2.width() - (s2.right - s1.left) * (s2.bottom - s1.top);
              iou = (s2.right - s1.left) * (s2.bottom - s1.top) / union;
              return iou;
          }
          else{
              union = s1.height() * s1.width() + s2.height() * s2.width() - (s2.right - s1.left) * (s1.bottom - s2.top);
              iou = (s2.right - s1.left) * (s1.bottom - s2.top) / union;
              return iou;
          }
      }
      else
          if (s1.top > s2.top){
              union = s1.height() * s1.width() + s2.height() * s2.width() - (s1.right - s2.left) * (s2.bottom - s1.top);
              iou = (s1.right - s2.left) * (s2.bottom - s1.top) / union;
              return iou;
          }

          else{
              union = s1.height() * s1.width() + s2.height() * s2.width() - (s1.right - s2.left) * (s1.bottom - s2.top);
              iou = (s1.right - s2.left) * (s1.bottom - s2.top) / union;
              return iou;
          }

  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
