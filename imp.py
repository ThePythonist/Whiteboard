import numpy as np
import cv2
import time
import sys
import pygame
from noise import pnoise1
from pygame.locals import *

def get_lines(image, ds=1, thickness=3, minLineLength=10, lineDist=300, smoothing=1000, smoothingRange=1):
    image = cv2.resize(image, (0,0), fx=1/ds, fy=1/ds)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, 100, 200)

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.linked = False

    class Line:
        def __init__(self, nodes):
            self.nodes = nodes
            self.ordered = False

        def __len__(self):
            return len(self.nodes)

        def dist(self, other):
            minDist = -1
            i = self.nodes[-1]
            for j in other.nodes:
                dist = abs(j.x-i.x) + abs(j.y-i.y)
                if minDist < 0 or dist < minDist:
                    minDist = dist
                if 0<= minDist <= 1:
                    return minDist
            return minDist


    grid = []
    nodes = []
    height, width = edges.shape
    for y in range(height):
        row = []
        for x in range(width):
            if edges[y, x] == 255:
                node = Node(x, y)
                row.append(node)
                nodes.append(node)
            else:
                row.append(None)
        grid.append(row)

    lines = []

    for startNode in nodes:
        if startNode.linked:
            continue
        line = []
        current = startNode
        finished = False
        while not finished:
            line.append(current)
            current.linked = True
            nextNode = None
            for r in range(1, thickness//2 + 1):
                top = max(0, current.y-r)
                bottom = min(height, current.y+r+1)
                left = max(0, current.x-r)
                right = min(width, current.x+r+1)
                for y in range(top, bottom):
                    for x in range(left, right):
                        if grid[y][x] is not None and not grid[y][x].linked:
                            nextNode = grid[y][x]
                            break
                    if nextNode is not None:
                        break
                if nextNode is not None:
                    break
            if nextNode is None:
                finished = True
            else:
                current = nextNode
        if len(line) > minLineLength:
            for i, l in enumerate(lines):
                if len(l) <= len(line):
                    lines = lines[:i] + [Line(line)] + lines[i:]
                    break
            else:
                lines.append(Line(line))
                
    orderedLines = []

    main = np.zeros((height, width))
    for startLine in lines:
        if startLine.ordered:
            continue
        current = startLine
        finished = False
        while not finished:
            orderedLines.append(current)
            current.ordered = True
            closest = None
            closestDist = -1
            for i in lines:
                if i.ordered:
                    continue
                dist = current.dist(i)
                if closest is None or dist < closestDist:
                    closest = i
                    closestDist = dist
            if closest is None:
                finished = True
            else:
                current = closest
    for i in orderedLines:
        for j, node in enumerate(i.nodes[smoothingRange:-smoothingRange]):
            startNode = i.nodes[j-smoothingRange]
            endNode = i.nodes[j+smoothingRange]

            x = endNode.x-startNode.x
            y = endNode.y-startNode.y

            e = node.x-startNode.x
            f = node.y-startNode.y

            mag = (x**2 + y**2)**2
            
            k = (f*x-e*y)/mag
            sign = -1 if k < 0 else 1
            l = sign*min(smoothing, abs(k))

            node.x += round(l * x/mag)
            node.y += round(l * y/mag)
            
        for j in i.nodes:
            j.x *= ds
            j.y *= ds
    return orderedLines

imagePath = sys.argv[1]
scale = 1
ds = 2
thickness = 10
noiseRoughness = 0.01
noiseAmplitude = 150
octaves = 8
persistence = 0.8
lacunarity = 0.4
pvScale = 1

image = cv2.imread(imagePath)
lines = get_lines(image, ds=ds, thickness=thickness)

height, width, *_ = image.shape

pygame.init()

pygame.display.set_mode((1, 1), 0, 32)

glare = pygame.image.load("effects/glare.png").convert_alpha()
glare = pygame.transform.smoothscale(glare, (glare.get_width()*scale,
                                             glare.get_height()*scale))

hand = pygame.image.load("effects/hand.png").convert_alpha()
hand = pygame.transform.scale(hand, (hand.get_width()*scale,
                                     hand.get_height()*scale))

screen = pygame.Surface((width*scale, height*scale))
screen.fill((250, 250, 255))

frameCount = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("video.mp4", fourcc, 20.0, (width*scale, height*scale))

def apply_effects(screen, handPos, doHand=True, relGlarePos=(0.75, 0.25)):
    global glare
    global hand
    surface = pygame.Surface((screen.get_width(), screen.get_height()))
    surface.blit(screen, (0, 0))
    glarePos = (relGlarePos[0]*surface.get_width()-glare.get_width()/2,
                relGlarePos[1]*surface.get_height()-glare.get_height()/2)
    surface.blit(glare, glarePos)
    if doHand:
        surface.blit(hand, handPos)
    return surface

def draw_link(screen, color, node, lastNode, thickness, scale, tail=True):
    pygame.draw.circle(screen, color, (node.x*scale, node.y*scale), (thickness*scale)//2)
    if lastNode is not None:
        if tail:
            pygame.draw.circle(screen, color, (lastNode.x*scale, lastNode.y*scale), (thickness*scale)//2)
        perp = [lastNode.y-node.y, node.x-lastNode.x]
        mag = (perp[0] ** 2 + perp[1] ** 2) ** 0.5
        perp[0] *= thickness/2/mag
        perp[1] *= thickness/2/mag
        pygame.draw.polygon(screen, color, (((lastNode.x-perp[0])*scale, (lastNode.y-perp[1])*scale),
                                            ((lastNode.x+perp[0])*scale, (lastNode.y+perp[1])*scale),
                                            ((node.x+perp[0])*scale, (node.y+perp[1])*scale),
                                            ((node.x-perp[0])*scale, (node.y-perp[1])*scale)))

preview = np.zeros((height, width)) + 255
for i in lines:
    for j in i.nodes:
        preview[j.y, j.x] = 0
cv2.imshow("preview", preview)

stopped = False
noiseX = 0
for line in lines:
    noiseX = int(noiseX) + 1
    lastNode = None
    for node in line.nodes:
        frameName = "frames/{}.png".format(frameCount)

        noise = max(0, min(1, (pnoise1(noiseX, octaves, persistence, lacunarity)+0.5)))*noiseAmplitude
        noiseX += noiseRoughness
        color = (noise, noise, noise)
        draw_link(screen, (200, 200, 200), node, lastNode, round(thickness+1), scale, False)
        draw_link(screen, color, node, lastNode, thickness, scale)
        
        lastNode = node
            
        pygame.image.save(apply_effects(screen, (node.x*scale, node.y*scale)), frameName)
        frame = cv2.imread(frameName)
        out.write(frame)

        scaledFrame = cv2.resize(frame, (0,0), fx=1/pvScale, fy=1/pvScale)
        cv2.imshow("video",scaledFrame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            stopped = True
            break
        
        frameCount += 1
    if stopped:
        break
pygame.image.save(apply_effects(screen, (node.x*scale, node.y*scale), False), "frames/end.png")
frame = cv2.imread("frames/end.png")
for i in range(50):
    out.write(frame)
print("Releasing")
out.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()

