from detect_balls_and_pockets import detect_only_pockets_and_draw , find_corner_pockets_from_mask
if __name__ == "__main__":
    print ("Starting test for detect_only_pockets_and_draw...")
    _, out = find_corner_pockets_from_mask('output/debug/final_detected.png')

    print ("Test completed.")
    print (out)

