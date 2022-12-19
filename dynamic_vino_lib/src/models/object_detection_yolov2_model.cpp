// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/** 
 * @brief a header file with declaration of ObjectDetectionModel class
 * @file object_detection_yolov2_model.cpp
 */

#include "dynamic_vino_lib/models/object_detection_yolov2_model.hpp"
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "dynamic_vino_lib/slog.hpp"
#include "dynamic_vino_lib/engines/engine.hpp"
#include "dynamic_vino_lib/inferences/object_detection.hpp"

// Validated Object Detection Network
Models::ObjectDetectionYolov2Model::ObjectDetectionYolov2Model(
  const std::string & label_loc, const std::string & model_loc, int max_batch_size)
: ObjectDetectionModel(label_loc, model_loc, max_batch_size)
{
}

bool Models::ObjectDetectionYolov2Model::updateLayerProperty(
  std::shared_ptr<ov::Model>& model)
{
  slog::info << "Checking INPUTs for model " << getModelName() << slog::endl;
  auto input_info_map = model->inputs();
  if (input_info_map.size() != 1) {
    slog::warn << "This model seems not Yolo-like, which has only one input, but we got "
      << std::to_string(input_info_map.size()) << "inputs" << slog::endl;
    return false;
  }
  // set input property
  ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
  input_tensor_name_ = model->input().get_any_name();
  ov::preprocess::InputInfo& input_info = ppp.input(input_tensor_name_);
  const ov::Layout input_tensor_layout{"NHWC"};
  input_info.tensor().
    set_element_type(ov::element::u8).
    set_layout(input_tensor_layout);
  addInputInfo("input", input_tensor_name_);


  // set output property
  auto output_info_map = model -> outputs();
  if (output_info_map.size() != 3) {
    slog::warn << "This model seems not Yolo-like! We got "
      << std::to_string(output_info_map.size()) << "outputs, but SSDnet has only one."
      << slog::endl;
    return false;
  }
  ov::preprocess::OutputInfo& output_info = ppp.output();
  addOutputInfo("output", model->output().get_any_name());
  output_info.tensor().set_element_type(ov::element::f32);
  slog::info << "Checking Object Detection output ... Name=" << model->output().get_any_name()
    << slog::endl;
  model = ppp.build();

#if(0) /// 
  const InferenceEngine::CNNLayerPtr output_layer =
    model->getNetwork().getLayerByName(output_info_map.begin()->first.c_str());
  // output layer should have attribute called num_classes
  slog::info << "Checking Object Detection num_classes" << slog::endl;
  if (output_layer == nullptr ||
    output_layer->params.find("classes") == output_layer->params.end()) {
    slog::warn << "This model's output layer (" << output_info_map.begin()->first
      << ") should have num_classes integer attribute" << slog::endl;
    return false;
  }
  // class number should be equal to size of label vector
  // if network has default "background" class, fake is used
  const int num_classes = output_layer->GetParamAsInt("classes");
  slog::info << "Checking Object Detection output ... num_classes=" << num_classes << slog::endl;
  if (getLabels().size() != num_classes) {
    if (getLabels().size() == (num_classes - 1)) {
      getLabels().insert(getLabels().begin(), "fake");
    } else {
      getLabels().clear();
    }
  }
#endif

  // last dimension of output layer should be 7
  auto outputsDataMap = model->outputs();
  auto & data = outputsDataMap[0];
  ov::Shape output_dims = data.get_shape();
  setMaxProposalCount(static_cast<int>(output_dims[2]));
  slog::info << "max proposal count is: " << getMaxProposalCount() << slog::endl;

  auto object_size = static_cast<int>(output_dims[3]);
  // if (object_size != 33) {
  //   slog::warn << "This model is NOT Yolo-like, whose output data for each detected object"
  //     << "should have 7 dimensions, but was " << std::to_string(object_size)
  //     << slog::endl;
  //   return false;
  // }
  setObjectSize(object_size);

  // if (output_dims.size() != 2) {
  //   slog::warn << "This model is not Yolo-like, output dimensions shoulld be 2, but was"
  //     << std::to_string(output_dims.size()) << slog::endl;
  //   return false;
  // }

  printAttribute();
  slog::info << "This model is Yolo-like, Layer Property updated!" << slog::endl;
  return true;
}

const std::string Models::ObjectDetectionYolov2Model::getModelCategory() const
{
  return "Object Detection Yolo v5";
}

bool Models::ObjectDetectionYolov2Model::enqueue(
  const std::shared_ptr<Engines::Engine> & engine,
  const cv::Mat & frame,
  const cv::Rect & input_frame_loc)
{
  setFrameSize(frame.cols, frame.rows);

  if (!matToBlob(frame, input_frame_loc, 1, 0, engine)) {
    return false;
  }
  return true;
}

bool Models::ObjectDetectionYolov2Model::matToBlob(
  const cv::Mat & orig_image, const cv::Rect &, float scale_factor,
  int batch_index, const std::shared_ptr<Engines::Engine> & engine)
{
  if (engine == nullptr) {
    slog::err << "A frame is trying to be enqueued in a NULL Engine." << slog::endl;
    return false;
  }

  std::string input_name = getInputName();
  ov::Tensor input_tensor =
    engine->getRequest().get_tensor(input_name);

  auto resized_image(orig_image);

  ov::Shape blob_size = input_tensor.get_shape();
  const int width = blob_size[3];
  const int height = blob_size[2];
  const int channels = blob_size[1];
  float * blob_data = input_tensor.data<float>();


  if(width != orig_image.size().width || height != orig_image.size().height)
  {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
  }

  // // TODO size is for demo
  size_t img_size = width*height;

  //nchw
  for(size_t row =0; row < height; row++)
  {
      for(size_t col=0; col < width; col++)
      {
          for(size_t ch =0; ch < channels; ch++)
          {
            blob_data[img_size*ch + row*width + col] = float(resized_image.at<cv::Vec3b>(row,col)[ch])/255.0f;
          }
      }
  }

  slog::debug << "Convert input image to blob: DONE!" << slog::endl;

  return true;
}

bool Models::ObjectDetectionYolov2Model::fetchResults(
  const std::shared_ptr<Engines::Engine> & engine,
  std::vector<dynamic_vino_lib::ObjectDetectionResult> & results,
  const float & confidence_thresh,
  const bool & enable_roi_constraint)
{
  slog::debug << "fetching Infer Resulsts from the given Yolov5 model" << slog::endl;

  if (engine == nullptr) {
    slog::err << "Trying to fetch results from <null> Engines." << slog::endl;
    return false;
  }

  ov::InferRequest request = engine->getRequest();

  std::string output = getOutputName();
  std::vector<std::string> & labels = getLabels();
  const float * detections = (float * )request.get_tensor(output).data();

  std::string input = getInputName();
  auto input_tensor = request.get_tensor(input);
  ov::Shape input_shape = input_tensor.get_shape();
  // int input_height = input_shape[2];
  // int input_width = input_shape[3];

  // --------------------------- Extracting layer parameters --------------------------------
  std::vector<cv::Rect>  origin_rect;
  std::vector<float> origin_rect_cof;
  int s[3] = {80,40,20};

  int index=0;

  for (auto &output : outputs_data_map_) 
  {
      auto output_name = output.first;
      ov::InferRequest request = engine->getRequest();
      ov::Tensor blob = request.get_tensor(output_name);
      parseYolov5(blob,s[index], confidence_thresh ,origin_rect, origin_rect_cof);
      ++index;
  }

  // TODO xiansen nms_area_threshold
  double nms_area_threshold = 0.02;

  //后处理获得最终检测结果
  std::vector<int> final_id;
  cv::dnn::NMSBoxes(origin_rect, origin_rect_cof,
                confidence_thresh ,nms_area_threshold,final_id);

    //根据final_id获取最终结果
    for(int i=0;i<final_id.size();++i)
    {
        cv::Rect resize_rect= origin_rect[final_id[i]];
        dynamic_vino_lib::ObjectDetectionResult result(resize_rect);
        result.setConfidence(origin_rect_cof[final_id[i]]);
        result.setLabel("");
        results.push_back(result);
    }

  slog::debug << "Analyzing YoloV5 Detection results..." << slog::endl;

  return true;
}

bool Models::ObjectDetectionYolov2Model::parseYolov5(const ov::Tensor &blob, int net_grid, float cof_threshold,
                            std::vector<cv::Rect>& o_rect, std::vector<float>& o_rect_cof)
{
  std::vector<int> anchors = getAnchors(net_grid);
  ov::InferRequest request = engine->getRequest();
  auto output_tensor = request.get_tensor(output);
  float * output_blob = output_tensor.data<float>();
   //80个类是85,一个类是6,n个类是n+5
   //int item_size = 6;
   int item_size = 85;
    size_t anchor_n = 3;
    for(int n=0;n<anchor_n;++n)
    {
      for(int i=0;i<net_grid;++i)
      {
        for(int j=0;j<net_grid;++j)
        {
            double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 4];
            box_prob = sigmoid(box_prob);
            //框置信度不满足则整体置信度不满足
            if(box_prob < cof_threshold)
                continue;
            
            //注意此处输出为中心点坐标,需要转化为角点坐标
            double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
            double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
            double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
            double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 3];
            
            double max_prob = 0;
            int idx=0;
            for(int t=5;t<85;++t){
                double tp= output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t];
                tp = sigmoid(tp);
                if(tp > max_prob){
                    max_prob = tp;
                    idx = t;
                }
            }
            float cof = box_prob * max_prob;                
            //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
            if(cof < cof_threshold)
                continue;

            x = (sigmoid(x)*2 - 0.5 + j)*640.0f/net_grid;
            y = (sigmoid(y)*2 - 0.5 + i)*640.0f/net_grid;
            w = pow(sigmoid(w)*2,2) * anchors[n*2];
            h = pow(sigmoid(h)*2,2) * anchors[n*2 + 1];

            double r_x = x - w/2;
            double r_y = y - h/2;
            cv::Rect rect(round(r_x),round(r_y),round(w),round(h));
            o_rect.push_back(rect);
            o_rect_cof.push_back(cof);
        }
      }
    }

    if(o_rect.size() == 0) 
    {
      return false;
    }

    return true;
}

//以下为工具函数
// int Models::ObjectDetectionYolov2Model::getEntryIndex(int side, int lcoords, int lclasses, int location, int entry)
// {
//   int n = location / (side * side);
//   int loc = location % (side * side);
//   return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
// }

double Models::ObjectDetectionYolov2Model::sigmoid(double x)
{
  return (1 / (1 + exp(-x)));
}

std::vector<int> Models::ObjectDetectionYolov2Model::getAnchors(int net_grid)
{
    std::vector<int> anchors(6);
    int a80[6] = {10,13, 16,30, 33,23};
    int a40[6] = {30,61, 62,45, 59,119};
    int a20[6] = {116,90, 156,198, 373,326}; 
    if(net_grid == 80)
    {
        anchors.insert(anchors.begin(),a80,a80 + 6);
    }
    else if(net_grid == 40)
    {
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20)
    {
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    return anchors;
}
