import colortransforms as ct
import Image

I = Image.open('../images/scanner/baguette/baguette1.tif')

color = I.getdata()[0] # any pixel

data = ct.rgb_to_cielab(255,0,0) # R G B

print data


