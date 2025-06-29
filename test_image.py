import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_image_with_faces():
    """Create a test image with multiple face-like patterns for testing"""
    
    # Create a 800x600 image with white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Convert to PIL for easier drawing
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw simple face patterns
    faces = [
        (150, 150, 100),  # x, y, size
        (400, 200, 80),
        (600, 300, 120),
        (200, 400, 90)
    ]
    
    for x, y, size in faces:
        # Face outline (circle)
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], 
                    outline=(0, 0, 0), width=3, fill=(255, 220, 177))
        
        # Eyes
        eye_offset = size // 4
        eye_size = size // 10
        draw.ellipse([x-eye_offset-eye_size, y-eye_offset-eye_size, 
                     x-eye_offset+eye_size, y-eye_offset+eye_size], 
                    fill=(0, 0, 0))
        draw.ellipse([x+eye_offset-eye_size, y-eye_offset-eye_size, 
                     x+eye_offset+eye_size, y-eye_offset+eye_size], 
                    fill=(0, 0, 0))
        
        # Nose (small line)
        draw.line([x, y, x, y+size//6], fill=(0, 0, 0), width=2)
        
        # Mouth (arc)
        mouth_y = y + size//4
        draw.arc([x-size//4, mouth_y-size//8, x+size//4, mouth_y+size//8], 
                0, 180, fill=(0, 0, 0), width=2)
    
    # Add text
    try:
        font = ImageFont.load_default()
        draw.text((50, 50), "Test Image - Multiple Faces", fill=(0, 0, 0), font=font)
        draw.text((50, 550), "Created for computer vision testing", fill=(0, 0, 0), font=font)
    except:
        pass
    
    # Convert back to numpy array
    result = np.array(pil_img)
    
    # Save the test image using PIL to avoid OpenCV dependency
    pil_img.save('test_faces.jpg')
    print("Test image saved as 'test_faces.jpg'")
    
    return result

if __name__ == "__main__":
    create_test_image_with_faces()
