//
//  GoogLeNetPlaces.m
//  CoreMLLearn
//
//  Created by liujizhou on 2017/6/10.
//  Copyright © 2017年 ZLJ. All rights reserved.
//

#import "GoogLeNetPlaces.h"

@implementation GoogLeNetPlacesInput

- (instancetype)initWithSceneImage:(CVPixelBufferRef)sceneImage {
    if (self) {
        _sceneImage = sceneImage;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"sceneImage"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"sceneImage"]) {
        return [MLFeatureValue featureValueWithPixelBuffer:_sceneImage];
    }
    return nil;
}

@end

@implementation GoogLeNetPlacesOutput


- (instancetype)initWithSceneLabelProbs:(NSDictionary<NSString *, NSNumber *> *)sceneLabelProbs sceneLabel:(NSString *)sceneLabel {
    if (self) {
        _sceneLabelProbs = sceneLabelProbs;
        _sceneLabel = sceneLabel;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"sceneLabelProbs", @"sceneLabel"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"sceneLabelProbs"]) {
        return [MLFeatureValue featureValueWithDictionary:_sceneLabelProbs error:nil];
    }
    if ([featureName isEqualToString:@"sceneLabel"]) {
        return [MLFeatureValue featureValueWithString:_sceneLabel];
    }
    return nil;
}

@end

@implementation GoogLeNetPlaces


- (nullable instancetype)initWithContentsOfURL:(NSURL *)url error:(NSError * _Nullable * _Nullable)error {
    self = [super init];
    if (!self) { return nil; }
    _model = [MLModel modelWithContentsOfURL:url error:error];
    if (_model == nil) { return nil; }
    return self;
}

- (nullable instancetype)init {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"GoogLeNetPlaces" ofType:@"mlmodelc"];
    return [self initWithContentsOfURL:[NSURL fileURLWithPath:assetPath] error:nil];
}

- (nullable GoogLeNetPlacesOutput *)predictionFromFeatures:(GoogLeNetPlacesInput *)input error:(NSError * _Nullable * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [_model predictionFromFeatures:input error:error];
    GoogLeNetPlacesOutput * result = [[GoogLeNetPlacesOutput alloc] initWithSceneLabelProbs:(NSDictionary<NSString *, NSNumber *> *)[outFeatures featureValueForName:@"sceneLabelProbs"].dictionaryValue sceneLabel:[outFeatures featureValueForName:@"sceneLabel"].stringValue];
    return result;
}

- (nullable GoogLeNetPlacesOutput *)predictionFromSceneImage:(CVPixelBufferRef)sceneImage error:(NSError * _Nullable * _Nullable)error {
    GoogLeNetPlacesInput *input_ = [[GoogLeNetPlacesInput alloc] initWithSceneImage:sceneImage];
    return [self predictionFromFeatures:input_ error:error];
}

@end

