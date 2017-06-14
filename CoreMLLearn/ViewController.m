//
//  ViewController.m
//  CoreMLLearn
//
//  Created by liujizhou on 2017/6/10.
//  Copyright © 2017年 ZLJ. All rights reserved.
//

#import "ViewController.h"
#import "GoogLeNetPlaces.h"
#import "UIImage+Utils.h"
#import "Resnet50.h"
#import "Inceptionv3.h"
#import "MNISTClassifier.h"
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>

@interface ViewController ()<UIImagePickerControllerDelegate, UINavigationControllerDelegate>

@property (nonatomic, strong) UILabel *resultLabel;
@property (nonatomic, strong) UIImageView *imageView;
@property (nonatomic, strong) UIImage *testImage;
@property (nonatomic, strong) CIImage* inputImage;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    UIImageView *originImgView = [[UIImageView alloc] initWithFrame:CGRectMake(0, 0, [UIScreen mainScreen].bounds.size.width, 400)];
    [self.view addSubview:originImgView];
    self.imageView = originImgView;
    
    UIButton *selectBtn = [[UIButton alloc] initWithFrame:CGRectMake(10, 410, 100, 50)];
    [self.view addSubview:selectBtn];
    [selectBtn addTarget:self action:@selector(localPhoto) forControlEvents:UIControlEventTouchUpInside];
    selectBtn.backgroundColor = [UIColor grayColor];
    [selectBtn setTitle:@"选择照片" forState:UIControlStateNormal];
    
    UIButton *imageRecognition = [[UIButton alloc] initWithFrame:CGRectMake(120, 410, 100, 50)];
    [self.view addSubview:imageRecognition];
    [imageRecognition addTarget:self action:@selector(imageRecognition) forControlEvents:UIControlEventTouchUpInside];
    imageRecognition.backgroundColor = [UIColor grayColor];
    [imageRecognition setTitle:@"物体分类" forState:UIControlStateNormal];
    
    UIButton *characterRecognition = [[UIButton alloc] initWithFrame:CGRectMake(230, 410, 160, 50)];
    [self.view addSubview:characterRecognition];
    [characterRecognition addTarget:self action:@selector(characterRecognition) forControlEvents:UIControlEventTouchUpInside];
    characterRecognition.backgroundColor = [UIColor grayColor];
    [characterRecognition setTitle:@"矩形区域内容识别" forState:UIControlStateNormal];
    
    
    UILabel *resultLabel = [[UILabel alloc] initWithFrame:CGRectMake(15, 420, [UIScreen mainScreen].bounds.size.width - 30, 300)];
    resultLabel.numberOfLines = 0;
    [self.view addSubview:resultLabel];
    self.resultLabel = resultLabel;
}

// 打开本地相册
-(void)localPhoto
{
    //本地相册不需要检查，因为UIImagePickerController会自动检查并提醒
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    picker.delegate = self;
    //设置选择后的图片可被编辑
    picker.allowsEditing = YES;
    [self presentViewController:picker animated:YES completion:nil];
}

// 识别图片里的物体
-(void)imageRecognition
{
    UIImage *image = self.testImage;
    UIImage *scaledImage = [image scaleToSize:CGSizeMake(224, 224)];
    CVPixelBufferRef buffer = [image pixelBufferFromCGImage:scaledImage];
    NSString* googleResult = [self predictGoogLeNetPlaces:buffer];
    NSString *ret50 = [self predictionWithResnet50:buffer];
    
    UIImage *inceptionImage = [image scaleToSize:CGSizeMake(299, 299)];
    NSString *inception = [self predictionWithInceptionV3: [image pixelBufferFromCGImage:inceptionImage]];
    NSString *result = [NSString stringWithFormat:@"GoogleLeNet:\n%@\n\nResnet50:\n%@\n\nInceptionV3:\n%@",googleResult,ret50,inception];
    self.resultLabel.text = result;
    
}

//  识别图片里的数字
-(void)characterRecognition
{
    [self predictMINISTClassifier:self.testImage];
}

//UIImagePickerControllerDelegate
- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingImage:(UIImage *)image editingInfo:(NSDictionary<NSString *,id> *)editingInfo
{
    self.testImage = image;
    self.imageView.image = image;
    [picker dismissViewControllerAnimated:YES completion:nil];
}

- (NSString*)predictGoogLeNetPlaces:(CVPixelBufferRef )buffer {
    GoogLeNetPlaces *model = [[GoogLeNetPlaces alloc] init];
    NSError *error;
    GoogLeNetPlacesInput *input = [[GoogLeNetPlacesInput alloc] initWithSceneImage:buffer];
    GoogLeNetPlacesOutput *output = [model predictionFromFeatures:input error:&error];
    if (error) {
        NSString *errInfo = error.description;
        return errInfo;
    } else {
        return [NSString stringWithFormat:@"识别结果:%@,匹配率:%.2f",output.sceneLabel, [[output.sceneLabelProbs valueForKey:output.sceneLabel]floatValue]];
    };
}

- (CGImagePropertyOrientation)cgImagePropertyOrientation:(UIImage*)image {
    switch (image.imageOrientation) {
        case UIImageOrientationUp:
            return kCGImagePropertyOrientationUp;
        case UIImageOrientationDown:
            return kCGImagePropertyOrientationDown;
        case UIImageOrientationLeft:
            return kCGImagePropertyOrientationLeft;
        case UIImageOrientationRight:
            return kCGImagePropertyOrientationRight;
        case UIImageOrientationUpMirrored:
            return kCGImagePropertyOrientationUpMirrored;
        case UIImageOrientationDownMirrored:
            return kCGImagePropertyOrientationDownMirrored;
        case UIImageOrientationLeftMirrored:
            return kCGImagePropertyOrientationLeftMirrored;
        case UIImageOrientationRightMirrored:
            return kCGImagePropertyOrientationRightMirrored;
        default:
            return kCGImagePropertyOrientationUp;
    }
}

- (NSString*)predictionWithResnet50:(CVPixelBufferRef )buffer
{
    NSError *modelLoadError = nil;
    NSURL *modelUrl = [NSURL URLWithString:[[NSBundle mainBundle] pathForResource:@"Resnet50" ofType:@"mlmodelc"]];
    Resnet50* resnet50 = [[Resnet50 alloc] initWithContentsOfURL:modelUrl error:&modelLoadError];
    
    NSError *predictionError = nil;
    Resnet50Output *resnet50Output = [resnet50 predictionFromImage:buffer error:&predictionError];
    if (predictionError) {
        return predictionError.description;
    } else {
        // resnet50Output.classLabelProbs sort
        return [NSString stringWithFormat:@"识别结果:%@,匹配率:%.2f",resnet50Output.classLabel, [[resnet50Output.classLabelProbs valueForKey:resnet50Output.classLabel]floatValue]];
    }
}

- (NSString*)predictionWithInceptionV3:(CVPixelBufferRef )buffer
{
    NSError *modelLoadError = nil;
    NSURL *modelUrl = [NSURL URLWithString:[[NSBundle mainBundle] pathForResource:@"Inceptionv3" ofType:@"mlmodelc"]];
    Inceptionv3* inceptionV3 = [[Inceptionv3 alloc] initWithContentsOfURL:modelUrl error:&modelLoadError];
    
    NSError *predictionError = nil;
    for (int i = 0; i< 100; i++) {
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            [inceptionV3 predictionFromImage:buffer error:nil];
        });
    };
    Inceptionv3Output *inceptionV3Output = [inceptionV3 predictionFromImage:buffer error:&predictionError];
    if (predictionError) {
        return predictionError.description;
    } else {
         return [NSString stringWithFormat:@"识别结果:%@,匹配率:%.2f",inceptionV3Output.classLabel, [[inceptionV3Output.classLabelProbs valueForKey:inceptionV3Output.classLabel]floatValue]];
    }
}

- (void)predictMINISTClassifier:(UIImage* )uiImage {
    CIImage *ciImage = [CIImage imageWithCGImage:uiImage.CGImage];
    CGImagePropertyOrientation orientation = [self cgImagePropertyOrientation:uiImage];
    self.inputImage = [ciImage imageByApplyingOrientation:orientation];
    
    VNDetectRectanglesRequest* rectanglesRequest = [[VNDetectRectanglesRequest alloc]initWithCompletionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        [self handleRectangles:request error:error];
    }];
    
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:uiImage.CGImage orientation:orientation options:nil];
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        NSError* error = nil;
        [handler performRequests:@[rectanglesRequest] error:&error];
    });
}

- (CGRect)scaledCGRect:(CGRect)rect toSize:(CGSize)size
{
    return CGRectMake(rect.origin.x * size.width, rect.origin.y * size.height, rect.size.width * size.width, rect.size.height*size.height);
}

- (CGPoint)scaledCGPoint:(CGPoint)point toSize:(CGSize)size
{
    return CGPointMake(point.x * size.width, point.y * size.height);
}

- (void)handleRectangles:(VNRequest*)request error:(NSError*)error {
    VNRectangleObservation *detectedRectangle = request.results.firstObject;
    CGSize imageSize = self.inputImage.extent.size;
    CGRect boundingBox = [self scaledCGRect:detectedRectangle.boundingBox toSize:imageSize];
    if (!CGRectContainsRect(self.inputImage.extent, boundingBox)) {
        NSLog(@"invalid detected rectangle");
        return;
    }
    CGPoint topLeft = [self scaledCGPoint:detectedRectangle.topLeft toSize:imageSize];
    CGPoint topRight = [self scaledCGPoint:detectedRectangle.topRight toSize:imageSize];
    CGPoint bottomLeft =[self scaledCGPoint:detectedRectangle.bottomLeft toSize:imageSize];
    CGPoint bottomRight = [self scaledCGPoint:detectedRectangle.bottomRight toSize:imageSize];
    CIImage *cropImage = [self.inputImage imageByCroppingToRect:boundingBox];
    NSDictionary *param = [NSDictionary dictionaryWithObjectsAndKeys:[CIVector vectorWithCGPoint:topLeft],@"inputTopLeft",[CIVector vectorWithCGPoint:topRight],@"inputTopRight",[CIVector vectorWithCGPoint:bottomLeft],@"inputBottomLeft",[CIVector vectorWithCGPoint:bottomRight],@"inputBottomRight", nil];
    CIImage* filterImage = [cropImage imageByApplyingFilter:@"CIPerspectiveCorrection" withInputParameters:param];
    filterImage = [filterImage imageByApplyingFilter:@"CIColorControls" withInputParameters:[NSDictionary dictionaryWithObjectsAndKeys:@(0),kCIInputSaturationKey,@(32),kCIInputContrastKey, nil]];
    filterImage = [filterImage imageByApplyingFilter:@"CIColorInvert" withInputParameters:nil];
    UIImage *correctedImage = [UIImage imageWithCIImage:filterImage];
    dispatch_async(dispatch_get_main_queue(), ^{
        self.imageView.image = correctedImage;
    });
    VNImageRequestHandler *vnImageRequestHandler = [[VNImageRequestHandler alloc] initWithCIImage:filterImage options:nil];
    
    MNISTClassifier *model = [MNISTClassifier new];
    VNCoreMLModel *vnCoreModel = [VNCoreMLModel modelForMLModel:model.model error:nil];
    VNCoreMLRequest *classificationRequest = [[VNCoreMLRequest alloc] initWithModel:vnCoreModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        VNClassificationObservation *best = request.results.firstObject;
        
        NSString* result  = [NSString stringWithFormat:@"识别结果:%@,匹配率:%.2f",best.identifier,best.confidence];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.resultLabel.text = result;
        });
    }];
    NSError *imageError = nil;
    [vnImageRequestHandler performRequests:@[classificationRequest] error:&imageError];
}
@end
