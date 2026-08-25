#!/usr/bin/env python3
from moms_apriltag import TagGenerator2, TagGenerator3
from matplotlib import pyplot as plt
import imageio

# Create TagGenerator2 instance for tag36h11 family
tg2 = TagGenerator2("tag36h11")

# Create TagGenerator3 instance for Circle49h12 family
tg3 = TagGenerator3("tagCircle49h12")


# Generate a specific tag ID (e.g., tag ID 0)
tag_id = 100

# Generate tag36h11 tag
tag = tg2.generate(tag_id)

# Display the tag
plt.imshow(tag, cmap="gray")
plt.title(f"AprilTag tag36h11 - ID: {tag_id}")
plt.axis('off')
plt.show()

# Save the tag as PNG
filename = f"./tag36h11_tag_{tag_id}.png"
imageio.imwrite(filename, tag)
print(f"Tag saved as {filename}")
