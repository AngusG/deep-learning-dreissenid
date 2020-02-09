import scaleapi

#client = scaleapi.ScaleClient('YOUR_SCALE_API_KEY')
#client = scaleapi.ScaleClient('test_c4362bc26016421392275cb8d93289d7')
client = scaleapi.ScaleClient('live_2ff4b5744abf4bf1912fee4e74f1e463')

resp = client.create_segmentannotation_task(
    callback_url='gallowaa@uoguelph.ca',
    instruction='Draw a tight polygon around the live **mussels** in the image. Ignore the empty white shells.',
    attachment_type='image',
    #attachment='https://zebra-mussel-segmentation.s3.us-east-2.amazonaws.com/GLNI_3537-3_2017-08-30_image-2_crop.jpg',
    #attachment='https://zebra-mussel-segmentation.s3.us-east-2.amazonaws.com/2019-land/GLNI_12-2_2019-07-09_image-1_crop.jpg',
    #attachment='https://zebra-mussel-segmentation.s3.us-east-2.amazonaws.com/2019-land/GLNI_3788-2_2019-06-17_image-3_crop.jpg',
    #attachment='https://zebra-mussel-segmentation.s3.us-east-2.amazonaws.com/2019-land/GLNI_456-1_2019-06-04_image-2_crop.jpg',
    attachment='https://zebra-mussel-segmentation.s3.us-east-2.amazonaws.com/Tripod/Lab_1352_2019-05-25_GoPro-1.jpg',
    labels=[
      'background',
      'mussel',
    ],
    allow_unlabeled=False
)
