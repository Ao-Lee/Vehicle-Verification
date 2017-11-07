from align import AlignDatabase, GetAlignFuncByBoundingBox

if __name__ == '__main__':
    
    source = 'E:\\DM\\Udacity\\Public Security\\test_source'
    target = 'E:\\DM\\Udacity\\Public Security\\test_target'
    
    source = 'E:\\DM\\Udacity\\Carvana\\Data\\origin'
    target = 'E:\\DM\\Udacity\\Carvana\\Data\\aligned'

    F = GetAlignFuncByBoundingBox(output_size=299)
    AlignDatabase(source, target, align_func=F)

    