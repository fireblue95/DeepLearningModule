#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

class BasicBlockImpl : public torch::nn::Module
{
public:
    BasicBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
    {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false));
        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
        relu1 = torch::nn::ReLU(torch::nn::ReLUOptions(true));

        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false));
        bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu1", relu1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = relu1->forward(x);
        x = conv2->forward(x);
        x = bn2->forward(x);
        return x;
    }

private:
    torch::nn::Conv2d conv1{nullptr},
        conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr},
        bn2{nullptr};
    torch::nn::ReLU relu1{nullptr};
};
TORCH_MODULE(BasicBlock);

class BlockImpl : public torch::nn::Module
{
public:
    BlockImpl(int64_t in_channels, std::vector<int> filters, int64_t kernel_size = 3, int64_t stride = 2, bool is_identity = true)
    {
        this->is_identity = is_identity;

        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, filters[0], 1).stride((is_identity) ? 1 : stride).bias(false));
        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(filters[0]));
        relu1 = torch::nn::ReLU(torch::nn::ReLUOptions(true));

        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], filters[1], kernel_size).stride(1).bias(false).padding(1));
        bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(filters[1]));
        relu2 = torch::nn::ReLU(torch::nn::ReLUOptions(true));

        conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[1], filters[2], 1).stride(1).bias(false));
        bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(filters[2]));

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu1", relu1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("relu2", relu2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);

        if (!is_identity)
        {
            conv1_shortcut = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, filters[2], 1).stride(stride).bias(false));
            bn1_shortcut = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(filters[2]));
            register_module("conv1_shortcut", conv1_shortcut);
            register_module("bn1_shortcut", bn1_shortcut);
        }

        relu_out = torch::nn::ReLU(torch::nn::ReLUOptions(true));
        register_module("relu_out", relu_out);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor y = conv1->forward(x);
        y = bn1->forward(y);
        y = relu1->forward(y);

        y = conv2->forward(y);
        y = bn2->forward(y);
        y = relu2->forward(y);

        y = conv3->forward(y);
        y = bn3->forward(y);

        // Shortcut
        if (!is_identity)
        {
            x = conv1_shortcut->forward(x);
            x = bn1_shortcut->forward(x);
        }
        x = x.add(y);
        x = relu_out(x);
        return x;
    }

private:
    bool is_identity;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::ReLU relu1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::ReLU relu2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::BatchNorm2d bn3{nullptr};

    torch::nn::Conv2d conv1_shortcut{nullptr};
    torch::nn::BatchNorm2d bn1_shortcut{nullptr};
    torch::nn::ReLU relu_out{nullptr};
};
TORCH_MODULE(Block);

class ResNetImpl : public torch::nn::Module
{
public:
    ResNetImpl(int64_t layer_num, int64_t class_num)
    {
        int out_channals = 0;
        this->class_num = class_num;
        std::vector<int> layer_type = {18, 34, 50, 101, 152};
        if (find(layer_type.begin(), layer_type.end(), layer_num) == layer_type.end())
            throw std::invalid_argument("Layer Number Option: 18, 34, 50, 101, 152");

        std::map<int, std::vector<int>> block_num = {
            {18, {2, 2, 2, 2}},
            {34, {3, 4, 6, 3}},
            {50, {3, 4, 6, 3}},
            {101, {3, 4, 23, 3}},
            {152, {3, 8, 36, 3}}};

        std::vector<int> b_num = block_num[layer_num];

        layer1 = torch::nn::Sequential();
        layer2 = torch::nn::Sequential();
        layer3 = torch::nn::Sequential();
        layer4 = torch::nn::Sequential();
        layer5 = torch::nn::Sequential();
        layer_final = torch::nn::Sequential();

        // Layer 1 start
        conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false));
        bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        register_module("conv", conv);
        register_module("bn", bn);
        register_module("relu", relu);
        register_module("maxpool", maxpool);

        layer1->push_back(conv);
        layer1->push_back(bn);
        layer1->push_back(relu);
        layer1->push_back(maxpool);

        register_module("layer1", layer1);
        // Layer 1 end

        if (layer_num == 18 || layer_num == 34)
        {
            layer2->push_back(BasicBlock(64, 64));
            for (int i = 1; i < b_num[0]; ++i)
                layer2->push_back(BasicBlock(64, 64));

            layer3->push_back(BasicBlock(64, 128, 2));
            for (int i = 1; i < b_num[1]; ++i)
                layer3->push_back(BasicBlock(128, 128));

            layer4->push_back(BasicBlock(128, 256, 2));
            for (int i = 1; i < b_num[2]; ++i)
                layer4->push_back(BasicBlock(256, 256));

            layer5->push_back(BasicBlock(256, 512, 2));
            for (int i = 1; i < b_num[3]; ++i)
                layer5->push_back(BasicBlock(512, 512));
            out_channals = 512;
        }
        else
        {
            layer2->push_back(Block(64, std::vector<int>{64, 64, 256}, 3, 1, false));
            for (int i = 1; i < b_num[0]; ++i)
                layer2->push_back(Block(256, std::vector<int>{64, 64, 256}));

            layer3->push_back(Block(256, std::vector<int>{128, 128, 512}, 3, 2, false));
            for (int i = 1; i < b_num[1]; ++i)
                layer3->push_back(Block(512, std::vector<int>{128, 128, 512}));

            layer4->push_back(Block(512, std::vector<int>{256, 256, 1024}, 3, 2, false));
            for (int i = 1; i < b_num[2]; ++i)
                layer4->push_back(Block(1024, std::vector<int>{256, 256, 1024}));

            layer5->push_back(Block(1024, std::vector<int>{512, 512, 2048}, 3, 2, false));
            for (int i = 1; i < b_num[3]; ++i)
                layer5->push_back(Block(2048, std::vector<int>{512, 512, 2048}));
            out_channals = 2048;
        }

        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);

        // Final Layer start
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        flatten = torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1));

        register_module("avgpool", avgpool);
        register_module("flatten", flatten);

        layer_final->push_back(avgpool);
        layer_final->push_back(flatten);
        // Final Layer end
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer_final->forward(x);

        fc = torch::nn::Linear(torch::nn::LinearOptions(x.sizes()[1], class_num));
        register_module("fc", fc);
        x = fc->forward(x);
        return x;
    }

private:
    int64_t class_num;
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr}, layer5{nullptr}, layer_final{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::AvgPool2d avgpool{nullptr};
    torch::nn::Flatten flatten{nullptr};
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(ResNet);

int main()
{
    ResNet model = ResNet(50, 3);
    torch::Tensor input = torch::randn({1, 3, 300, 300});
    torch::Tensor output = model->forward(input);
    std::cout << output.sizes() << std::endl;

    return 0;
}