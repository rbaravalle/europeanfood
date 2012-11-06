import colortransforms as ct
import Image

I = Image.open('../images/scanner/baguette/baguette1.tif')

color = I.getdata()[0] # any pixel

data = ct.rgb_to_cielab(color[0], color[1], color[2]) # R G B

print data


