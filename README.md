# Bibnumber: automatic bib number recognition from racing photos

Bibnumber automatically recognizes bib number from racing photos. As an example, consider this photo:
![Example](samples/0017-IMG_0035.JPG)

Calling bibnumber on this example will produce the following output:

     [ 38 46 54 69 164 773 775]

Bibnumber is tuned for high accuracy rather than high recall. Therefore it is not unusual for a bib to be missing in the output however wrong bib numbers are unusual.
