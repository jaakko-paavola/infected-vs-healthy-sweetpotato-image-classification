
        ROI_original = original_image[y:y+h, x:x+w]
        plant_mask = mask_plant_parts(ROI_masked, return_only_mask=True)
        single_plant_masked = cv2.bitwise_and(ROI_masked, ROI_masked, mask=plant_mask).copy()
        single_plant_original = cv2.bitwise_and(ROI_original, ROI_original, mask=plant_mask).copy()

        # Write masked single plant
        masked_segmented_path = os.path.join(pathname, f"{masked_filename}_M{contour_index}{masked_filetype}")
        masked_segmented_paths.append(masked_segmented_path)
        cv2.imwrite(masked_segmented_path, single_plant_masked)

        # Write non-masked single plant
        original_segmented_path = os.path.join(pathname, f"{original_filename}_O{contour_index}{original_filetype}")
        original_segmented_paths.append(original_segmented_path)
        cv2.imwrite(original_segmented_path, single_plant_original)


    # for now, save the original image in the same location as the segments, just for easy checking that the segmentation has gone right
    cv2.imwrite(os.path.join(pathname, f"{masked_filename}{masked_filetype}"), masked_image)
    cv2.imwrite(os.path.join(pathname, f"{original_filename}{original_filetype}"), original_image)

    return masked_segmented_paths, original_segmented_paths

# %%

def test_segmentation():
    regex_pattern_for_plant_info = r".*([1-9]{2})-([1-9]).*(Tray_.*?)-"
    regex_pattern_for_extracting_filename_from_path = '.*\/(.*)$'

    for idx1, filename in enumerate(glob.glob(f'{input_path_of_tray_images}/*.png')):
        img = cv2.imread(filename)
        file = re.search(regex_pattern_for_extracting_filename_from_path, filename).group(1)
        prefix = re.match(regex_pattern_for_plant_info, filename).group(1)
        stage = re.match(regex_pattern_for_plant_info, filename).group(2)
        tray_id = re.match(regex_pattern_for_plant_info, filename).group(3)
        subfolder_name = f'{prefix}-{stage}-PS_{tray_id}'

        if(not os.path.exists(f'{output_path_for_separated_plants}/{subfolder_name}')):
            pathlib.Path(f'{output_path_for_separated_plants}/{subfolder_name}').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'{output_path_for_separated_plants}/{subfolder_name}/' + file, img)

        cnts = find_contours(img)
        sorted_cnts = sorted(cnts, key=cmp_to_key(contour_sort))
        for idx2, c in enumerate(sorted_cnts):
            x, y, w, h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            masked = mask_plant_parts(ROI)
            result = masked.copy()

            ## Make the background transparent (or comment out the following block to leave the (black) background)
            # gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            # ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # kernel = np.ones((9,9), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            # result[:, :, 3] = mask

            # Write the split plant in a file
            cv2.imwrite(f'{output_path_for_separated_plants}/{subfolder_name}/plant_index_{idx2+1}.png', result)

    # %%

    gc_df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, 'growth_chamber_plant_data.csv'))

    output_path_for_separated_plants_gc = '/Separated_plants/Trial_02/Dataset_03/Background_included'

    regex_pattern_for_plant_info_gc = r"^([0-9]{6}) - ([0-9]{2}) - TV - (Hua|R3)-(H|FMV|CSV|VD) - ((?:[0-9]{2}-|)[0-9]{2}) - Mask.png$"

    split_gc_df = pd.DataFrame(columns=['Trial', 'Dataset', 'Genotype', 'Condition', 'Original image path', 'Masked image path', 'Split masked image path'])

    for index, row in gc_df.iterrows():
        filename = row['Masked image path']
        img = cv2.imread(os.path.join(DATA_FOLDER_PATH, filename))
        file = re.search(regex_pattern_for_extracting_filename_from_path, filename).group(1)
        prefix = re.match(regex_pattern_for_plant_info_gc, file).group(1)
        tray = re.match(regex_pattern_for_plant_info_gc, file).group(2)
        genotype = re.match(regex_pattern_for_plant_info_gc, file).group(3)
        condition = re.match(regex_pattern_for_plant_info_gc, file).group(4)
        plants = re.match(regex_pattern_for_plant_info_gc, file).group(5)
        subfolder_name = f'{prefix} - {tray} - TV - {genotype}-{condition} - {plants}'
        if (not os.path.exists(f'{DATA_FOLDER_PATH}/{output_path_for_separated_plants_gc}/{subfolder_name}')):
            pathlib.Path(f'{DATA_FOLDER_PATH}/{output_path_for_separated_plants_gc}/{subfolder_name}').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'{DATA_FOLDER_PATH}/{output_path_for_separated_plants_gc}/{subfolder_name}/' + file, img)

        cnts = find_contours(img)
        sorted_cnts = sorted(cnts, key=cmp_to_key(contour_sort))
        for idx2, c in enumerate(sorted_cnts):
            x, y, w, h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            masked = mask_plant_parts(ROI)
            result = masked.copy()

            # Write the split plant in a file
            split_plant_img_path = f'{output_path_for_separated_plants_gc}/{subfolder_name}/plant_index_{idx2+1}.png'
            cv2.imwrite(f'{DATA_FOLDER_PATH}/{split_plant_img_path}', result)

            # Add an entry for the split plant to the new dataframe
            split_plant_data = {
                'Trial': row['Trial'],
                'Dataset': row['Dataset'],
                'Genotype': row['Genotype'],
                'Condition': row['Condition'],
                'Original image path': row['Original image path'],
                'Masked image path': row['Masked image path'],
                'Split masked image path': split_plant_img_path,
            }
            split_gc_df = split_gc_df.append(split_plant_data, ignore_index=True)

    # %%

    # There were two errors that needed manual work.
    # In both cases one of the plants in the group image was split into two separate images
    # for a single plant, as the leaves of the plant had too much gap between them.
    # Here I drop the automatically created unnecessary rows after manually merging and deleting
    # the separate images for these two plants.

    split_gc_df = pd.read_csv(f'{DATA_FOLDER_PATH}/growth_chamber_plant_data_split.csv')

    split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 05 - TV - R3-H - 14-15 - Mask')]['Split masked image path']
    split_gc_df.iloc[96]['Split masked image path']
    split_gc_df.drop(96, inplace=True)
    split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 05 - TV - R3-H - 14-15 - Mask')]['Split masked image path']

    split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 06 - TV - R3-FMV - 11-13 - Mask')]['Split masked image path']
    split_gc_df.iloc[110]['Split masked image path']
    split_gc_df.drop(110, inplace=True)
    split_gc_df[split_gc_df['Masked image path'].str.contains('180724 - 06 - TV - R3-FMV - 11-13 - Mask')]['Split masked image path']

    # %%

    # Drop extra slash from the beginning of the split masked image path name
    split_gc_df['Split masked image path'] = split_gc_df['Split masked image path'].str[1:]

    # %%

    split_gc_df.to_csv(f'{DATA_FOLDER_PATH}/growth_chamber_plant_data_split.csv')
