//
//  GoogLeNetPlaces.h
//  CoreMLLearn
//
//  Created by liujizhou on 2017/6/10.
//  Copyright © 2017年 ZLJ. All rights reserved.
//

#import <Foundation/Foundation.h>

#import <CoreML/CoreML.h>
#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

/// Model Prediction Input Type
@interface GoogLeNetPlacesInput : NSObject<MLFeatureProvider>
/// Input image of scene to be classified as RGB image buffer, 224 pixels wide by 224 pixels high
@property (readwrite, nonatomic) CVPixelBufferRef sceneImage;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithSceneImage:(CVPixelBufferRef)sceneImage;
@end

/// Model Prediction Output Type
@interface GoogLeNetPlacesOutput : NSObject<MLFeatureProvider>
/// Probability of each scene as dictionary of strings to doubles
@property (readwrite, nonatomic) NSDictionary<NSString *, NSNumber *> * sceneLabelProbs;
/// Most likely scene label as string value
@property (readwrite, nonatomic) NSString * sceneLabel;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithSceneLabelProbs:(NSDictionary<NSString *, NSNumber *> *)sceneLabelProbs sceneLabel:(NSString *)sceneLabel;
@end

/// Class for model loading and prediction
@interface GoogLeNetPlaces : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;
- (nullable instancetype)initWithContentsOfURL:(NSURL *)url error:(NSError * _Nullable * _Nullable)error;
/// Make a prediction using the standard interface
/// @param input an instance of GoogLeNetPlacesInput to predict from
/// @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
/// @return the prediction as GoogLeNetPlacesOutput
- (nullable GoogLeNetPlacesOutput *)predictionFromFeatures:(GoogLeNetPlacesInput *)input error:(NSError * _Nullable * _Nullable)error;
/// Make a prediction using the convenience interface
/// @param sceneImage Input image of scene to be classified as RGB image buffer, 224 pixels wide by 224 pixels high:
/// @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
/// @return the prediction as GoogLeNetPlacesOutput
- (nullable GoogLeNetPlacesOutput *)predictionFromSceneImage:(CVPixelBufferRef)sceneImage error:(NSError * _Nullable * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
