// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/ten_crop.h"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {
namespace cpu {

Result<Tensor> CropImage(Stream& stream, const Device& device, const Tensor& tensor, int top,
                         int left, int bottom, int right);

class TenCropImpl : public ::mmdeploy::TenCropImpl {
 public:
  explicit TenCropImpl(const Value& args) : ::mmdeploy::TenCropImpl(args) {}

 protected:
  Result<Tensor> CropImage(const Tensor& tensor, int top, int left, int bottom,
                           int right) override {
    return ::mmdeploy::cpu::CropImage(stream_, device_, tensor, top, left, bottom, right);
  }

  Result<Tensor> HorizontalFlip(const Tensor& tensor) {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));
    SyncOnScopeExit(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);
    cv::Mat mat = Tensor2CVMat(src_tensor);
    cv::Mat flipped_mat;
    cv::flip(mat, flipped_mat, 1);
    return CVMat2Tensor(flipped_mat);
  }
};

class TenCropImplCreator : public Creator<::mmdeploy::TenCropImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<TenCropImpl>(args); }
};

}  // namespace cpu
}  // namespace mmdeploy

using ::mmdeploy::TenCropImpl;
using ::mmdeploy::cpu::TenCropImplCreator;
REGISTER_MODULE(TenCropImpl, TenCropImplCreator);
