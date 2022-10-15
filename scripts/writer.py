from PIL import ImageFont, Image, ImageDraw

def text_wrap(text, font, max_width):
    lines = []
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        words = text.split(' ')  
        i = 0
        while i < len(words):
            line = ''        
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:                
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)   
    return lines
 

def draw_text(text, org, size):    
  
    img = Image.open('./results/image.jpg')
    draw = ImageDraw.Draw(img)
    font_file_path = r'./fonts/timesnewromanbolditalic.ttf'
    font = ImageFont.truetype(font_file_path, size=int(size*3/4))
    lines = text_wrap(text, font, org[0])
    line_height = font.getsize('hg')[1]
    color = (255, 255, 255)
    for line in lines:
        draw.text((org[0], org[1] - line_height), text=line, color = color, font=font)
        org[1] = org[1] + line_height
        
    # img.show()
    img.save('final.jpg', optimize=True)



