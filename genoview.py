from pyray import (
    Vector2, Vector3, Vector4, Transform, Matrix, Camera3D, 
    Color, Rectangle, Model, ModelAnimation, Mesh, BoneInfo, 
    Texture, RenderTexture)
from raylib import *

import bvh
import quat
import numpy as np
import struct
import os
import sys
import subprocess
import cffi
ffi = cffi.FFI()

#----------------------------------------------------------------------------------
# Camera
#----------------------------------------------------------------------------------

class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()
    
    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance +  20.0 * dt * -mouseWheel, 0.1, 100.0)
        
        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye        

#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:
    
    def __init__(self):
        
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0
    
    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target

def UnloadShadowMap(target):
    
    if target.id > 0:
        rlUnloadFramebuffer(target.id)
        

def BeginShadowMap(target, shadowLight):
    
    BeginTextureMode(target)
    ClearBackground(WHITE)
    
    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2, 
        -shadowLight.height/2, shadowLight.height/2, 
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D    


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()

def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)

#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:
    
    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture 
        self.normal = Texture()  # Normal buffer attachment texture 
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):
    
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2) 

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    # Use raylib's own camera setup for the projection/view matrices. The
    # hand-rolled rlFrustum + rlMultMatrixf path silently culls all geometry
    # on macOS's OpenGL driver, so we let BeginMode3D build the matrices
    # (it pushes the projection matrix; EndMode3D pops it in EndGBuffer).
    BeginMode3D(camera)


def EndGBuffer(windowWidth, windowHeight):

    EndMode3D()                     # Restore matrices (pairs with BeginMode3D in BeginGBuffer)

    rlActiveDrawBuffers(1)
    rlDisableFramebuffer()          # Disable render target (fbo)


#----------------------------------------------------------------------------------
# Geno Character and Animation
#----------------------------------------------------------------------------------

def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)

def LoadGenoModel(fileName):

    model = Model()
    model.transform = MatrixIdentity()
  
    with open(fileName, "rb") as f:
        
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * ffi.sizeof(Mesh()))
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * ffi.sizeof("float"))
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("float"))
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ffi.sizeof("unsigned short"))
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.bones =  MemAlloc(model.boneCount * ffi.sizeof(BoneInfo()))
        model.bindPose =  MemAlloc(model.boneCount * ffi.sizeof(Transform()))
        
        FileRead(model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, ffi.sizeof("float") * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, ffi.sizeof("unsigned char") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, ffi.sizeof("float") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, ffi.sizeof("unsigned short") * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, ffi.sizeof(BoneInfo()) * model.boneCount, f)
        FileRead(model.bindPose, ffi.sizeof(Transform()) * model.boneCount, f)
        
        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * ffi.sizeof(Matrix()))
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()
    
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    
    return model


def GetModelBindPoseAsNumpyArrays(model):
    
    bindPos = np.zeros([model.boneCount, 3])
    bindRot = np.zeros([model.boneCount, 4])
    
    for boneId in range(model.boneCount):
        bindTransform = model.bindPose[boneId]
        bindPos[boneId] = (bindTransform.translation.x, bindTransform.translation.y, bindTransform.translation.z)
        bindRot[boneId] = (bindTransform.rotation.w, bindTransform.rotation.x, bindTransform.rotation.y, bindTransform.rotation.z)
        
    return bindPos, bindRot
    
    
def UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot, animPos, animRot):
    
    meshPos = quat.mul_vec(animRot, quat.inv_mul_vec(bindRot, -bindPos)) + animPos
    meshRot = quat.mul_inv(animRot, bindRot)
    
    matArray = np.frombuffer(ffi.buffer(
        model.meshes[0].boneMatrices, model.boneCount * 4 * 4 * 4), 
        dtype=np.float32).reshape([model.boneCount, 4, 4])
    
    matArray[:,:3,:3] = quat.to_xform(meshRot)
    matArray[:,:3,3] = meshPos


#----------------------------------------------------------------------------------
# Debug Draw
#----------------------------------------------------------------------------------

def DrawTransform(position, rotation, scale):
    
    rotMatrix = QuaternionToMatrix(Vector4(*rotation))
  
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2)),
        RED)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6)),
        GREEN)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10)),
        BLUE)

def DrawSkeleton(positions, rotations, parents, color):
    
    for i in range(len(positions)):
    
        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        DrawTransform(positions[i], rotations[i], 0.1)

        if parents[i] != -1:

            DrawLine3D(
                Vector3(*positions[i]),
                Vector3(*positions[parents[i]]),
                color)


def DrawSkeletonOverlay(positions, parents):
    # Clean skeleton: solid bones (joint -> parent) with joint markers, drawn as
    # an overlay so it reads clearly over the skinned mesh.
    boneColor = Color(235, 235, 240, 255)
    jointColor = Color(255, 190, 40, 255)
    for i in range(len(positions)):
        par = parents[i]
        if par != -1:
            DrawCylinderEx(Vector3(*positions[par]), Vector3(*positions[i]),
                           0.007, 0.007, 6, boneColor)
    for i in range(len(positions)):
        DrawSphere(Vector3(*positions[i]), 0.013, jointColor)


#----------------------------------------------------------------------------------
# Foot Slide Detection
#----------------------------------------------------------------------------------

# A foot is "planted" only when its contact joint stays close to the ground for
# a sustained window of frames - not just for the instant it passes low through
# the bottom of a step. This is what separates a real foot SLIDE (a planted foot
# that drifts) from a foot that is simply moving/stepping near the ground.

FOOT_CONTACT_BAND   = 0.03   # m above the joint's lowest point that counts as touching the ground
FOOT_CONTACT_FRAMES = 5      # +/- frames the foot must stay grounded to count as planted (~contact persistence)
FOOT_SLIDE_SPEED    = 0.10   # m/s: a planted foot moving faster than this is flagged as sliding
FOOT_SLIDE_FULL     = 0.30   # m/s: slide speed mapped to a fully-red marker
FOOT_FLOOR_LEVEL    = -0.02  # m: a foot joint below this has penetrated the floor (small tolerance)


def ComputeFootContacts(globalPositions, jointIndex, dt):
    # Pre-computes, for one foot joint over the whole clip:
    #   hSpeed[f] - horizontal speed (m/s)
    #   planted[f] - sustained ground contact (grounded across a centred window)
    p = globalPositions[:, jointIndex, :]
    h = p[:, 1]
    hSpeed = np.empty(len(h), np.float32)
    hSpeed[0] = 0.0
    hSpeed[1:] = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) / dt

    groundedH = h < (h.min() + FOOT_CONTACT_BAND)
    # planted[f] = groundedH for every frame in [f - W, f + W]  (via prefix sums)
    W = FOOT_CONTACT_FRAMES
    csum = np.concatenate([[0], np.cumsum(groundedH.astype(np.int32))])
    F = len(h)
    planted = np.zeros(F, bool)
    for f in range(F):
        a = max(0, f - W); b = min(F, f + W + 1)
        planted[f] = (csum[b] - csum[a]) == (b - a)
    return hSpeed, planted


def MeasureFootSlide(globalPositions, frame, jointIndex, contacts):
    # contacts = (hSpeedArray, plantedArray) from ComputeFootContacts
    hSpeedArr, plantedArr = contacts
    f = min(frame, len(hSpeedArr) - 1)
    hSpeed = float(hSpeedArr[f])
    grounded = bool(plantedArr[f])
    sliding = grounded and (hSpeed > FOOT_SLIDE_SPEED)
    p = globalPositions[frame][jointIndex]
    penetrating = float(p[1]) < FOOT_FLOOR_LEVEL
    return Vector3(float(p[0]), float(p[1]), float(p[2])), hSpeed, grounded, sliding, penetrating


def DrawFootSlide(pos, hSpeed, grounded, sliding, penetrating):
    if penetrating:
        # foot is below the floor - flag it (magenta), regardless of slide state
        DrawSphere(pos, 0.028, MAGENTA)
        DrawLine3D(pos, Vector3(pos.x, 0.0, pos.z), MAGENTA)
        DrawCircle3D(Vector3(pos.x, 0.001, pos.z), 0.06, Vector3(1.0, 0.0, 0.0), 90.0, MAGENTA)
    elif grounded:
        # green (planted, still) -> red, ramping only once it is actually sliding
        t = max(0.0, min(1.0, (hSpeed - FOOT_SLIDE_SPEED) / (FOOT_SLIDE_FULL - FOOT_SLIDE_SPEED)))
        col = Color(int(40 + 215 * t), int(200 * (1.0 - t) + 40 * t), 55, 255)
        DrawSphere(pos, 0.025, col)
        DrawLine3D(pos, Vector3(pos.x, 0.0, pos.z), col)   # drop line to the floor
        if sliding:
            # small flat ring on the floor marking the sliding contact
            r = min(0.07, 0.035 + hSpeed * 0.06)
            DrawCircle3D(Vector3(pos.x, 0.001, pos.z), r, Vector3(1.0, 0.0, 0.0), 90.0, RED)
    else:
        DrawSphere(pos, 0.018, Fade(SKYBLUE, 0.45))        # airborne: faint blue


#----------------------------------------------------------------------------------
# Onion Skinning (ghost frames)
#----------------------------------------------------------------------------------

# Draw the pose at frame +/- k*stride as translucent "ghosts" so the motion is
# legible in a still frame. Past frames are tinted cool, future frames warm, and
# both fade out the further they are in time. Drawn as a forward translucent
# overlay (the deferred G-buffer can only hold one opaque surface per pixel).

ONION_STRIDE       = 6      # frames between successive ghosts
ONION_COUNT        = 3      # number of ghosts per side (past / future)
ONION_MAX_ALPHA    = 0.32   # opacity of the nearest ghost
ONION_PAST_COLOR   = (0.35, 0.55, 0.95)   # cool blue  - earlier frames
ONION_FUTURE_COLOR = (0.95, 0.50, 0.25)   # warm orange - later frames


def DrawOnionGhosts(model, bindPos, bindRot, globalPositions, globalRotations,
                    frame, maxFrame, ghostShader, ghostColorLoc):
    model.materials[0].shader = ghostShader
    colPtr = ffi.new('float[4]')
    rlEnableColorBlend()
    for side, base in ((-1, ONION_PAST_COLOR), (1, ONION_FUTURE_COLOR)):
        for k in range(1, ONION_COUNT + 1):
            fg = frame + side * k * ONION_STRIDE
            if fg < 0 or fg > maxFrame:
                continue
            alpha = ONION_MAX_ALPHA * (1.0 - (k - 1) / float(ONION_COUNT))
            colPtr[0], colPtr[1], colPtr[2], colPtr[3] = base[0], base[1], base[2], alpha
            SetShaderValue(ghostShader, ghostColorLoc, colPtr, SHADER_UNIFORM_VEC4)
            UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot,
                                           globalPositions[fg], globalRotations[fg])
            DrawModel(model, Vector3Zero(), 1.0, WHITE)
    rlDisableColorBlend()
    # restore the model to the current frame's pose for subsequent passes
    UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot,
                                   globalPositions[frame], globalRotations[frame])


#----------------------------------------------------------------------------------
# Animation loading (supports runtime swapping / hot-reload)
#----------------------------------------------------------------------------------

def LoadAnimation(path):
    # Loads a BVH and pre-computes everything the viewer needs for it. Returns a
    # dict so the same routine can be used at start-up and for runtime reloads.
    bvhData = bvh.load(path)
    parents = bvhData['parents']
    localPositions = 0.01 * bvhData['positions'].copy().astype(np.float32)
    localRotations = quat.unroll(quat.from_euler(np.radians(bvhData['rotations']), order=bvhData['order']))
    globalRotations, globalPositions = quat.fk(localRotations, localPositions, parents)

    frameTime = bvhData.get('frametime', 1.0 / 60.0)
    if frameTime <= 0.0:
        frameTime = 1.0 / 60.0

    names = bvhData['names']
    footSpec = [("L Toe", "LeftToeBase"), ("R Toe", "RightToeBase"),
                ("L Foot", "LeftFoot"), ("R Foot", "RightFoot")]
    footJoints = [(label, names.index(name)) for (label, name) in footSpec if name in names]
    # foot speeds are in m/s, so use the clip's real frame time
    footContacts = {j: ComputeFootContacts(globalPositions, j, frameTime) for (_, j) in footJoints}

    return {
        'path': path,
        'name': os.path.basename(path),
        'parents': parents,
        'localPositions': localPositions,
        'globalRotations': globalRotations,
        'globalPositions': globalPositions,
        'footJoints': footJoints,
        'footContacts': footContacts,
        'frameTime': frameTime,
    }


def ScanBvhFolder(folder):
    # Returns a sorted list of full .bvh paths in the folder (empty if missing).
    try:
        return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith(".bvh")]
    except OSError:
        return []


def NativeChoose(folder=False):
    # Opens a native macOS file/folder picker via osascript and returns the chosen
    # POSIX path (or None if cancelled / unsupported). Used by the Open buttons.
    if sys.platform != "darwin":
        print("Native file picker is only wired up for macOS; use drag-and-drop instead.")
        return None
    # NOTE: we deliberately don't pass `of type {"bvh"}` - .bvh has no registered
    # UTI on macOS, which makes the dialog grey out every file. Instead we let any
    # file be chosen and validate the extension afterwards.
    if folder:
        script = 'POSIX path of (choose folder with prompt "Select a folder of BVH files")'
    else:
        script = 'POSIX path of (choose file with prompt "Select a BVH file")'
    try:
        out = subprocess.run(["osascript", "-e", script],
                             capture_output=True, text=True, timeout=300)
        path = out.stdout.strip()
        return path or None
    except Exception as e:
        print("File picker failed:", e)
        return None


#----------------------------------------------------------------------------------
# Shader loading
#----------------------------------------------------------------------------------

def LoadShaderVersioned(vsPath, fsPath):
    # Loads shaders while adapting the GLSL "#version" line to the GL backend
    # raylib was built against: desktop GL (>= 3.3) wants "#version 330", while
    # web / mobile want "#version 300 es". The shader sources stay written as
    # GLSL ES so they work on every platform; only the version line is rewritten.
    header = "#version 330"
    if rlGetVersion() in (RL_OPENGL_ES_20, RL_OPENGL_ES_30):
        header = "#version 300 es"

    def patched(path):
        with open(path) as f:
            lines = f.read().split("\n")
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#version"):
                lines[i] = header
                break
        return "\n".join(lines).encode()

    return LoadShaderFromMemory(patched(vsPath), patched(fsPath))


#----------------------------------------------------------------------------------
# App
#----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Init Window
    
    screenWidth = 1280
    screenHeight = 720
    
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(screenWidth, screenHeight, b"GenoViewPython")
    SetTargetFPS(60)

    # Shaders
    
    shadowShader = LoadShaderVersioned("./resources/shadow.vs", "./resources/shadow.fs")
    shadowShaderLightClipNear = GetShaderLocation(shadowShader, b"lightClipNear")
    shadowShaderLightClipFar = GetShaderLocation(shadowShader, b"lightClipFar")
    
    skinnedShadowShader = LoadShaderVersioned("./resources/skinnedShadow.vs", "./resources/shadow.fs")
    skinnedShadowShaderLightClipNear = GetShaderLocation(skinnedShadowShader, b"lightClipNear")
    skinnedShadowShaderLightClipFar = GetShaderLocation(skinnedShadowShader, b"lightClipFar")
    
    skinnedBasicShader = LoadShaderVersioned("./resources/skinnedBasic.vs", "./resources/basic.fs")
    skinnedBasicShaderSpecularity = GetShaderLocation(skinnedBasicShader, b"specularity")
    skinnedBasicShaderGlossiness = GetShaderLocation(skinnedBasicShader, b"glossiness")
    skinnedBasicShaderCamClipNear = GetShaderLocation(skinnedBasicShader, b"camClipNear")
    skinnedBasicShaderCamClipFar = GetShaderLocation(skinnedBasicShader, b"camClipFar")

    basicShader = LoadShaderVersioned("./resources/basic.vs", "./resources/basic.fs")
    basicShaderSpecularity = GetShaderLocation(basicShader, b"specularity")
    basicShaderGlossiness = GetShaderLocation(basicShader, b"glossiness")
    basicShaderCamClipNear = GetShaderLocation(basicShader, b"camClipNear")
    basicShaderCamClipFar = GetShaderLocation(basicShader, b"camClipFar")
    
    lightingShader = LoadShaderVersioned("./resources/post.vs", "./resources/lighting.fs")
    lightingShaderGBufferColor = GetShaderLocation(lightingShader, b"gbufferColor")
    lightingShaderGBufferNormal = GetShaderLocation(lightingShader, b"gbufferNormal")
    lightingShaderGBufferDepth = GetShaderLocation(lightingShader, b"gbufferDepth")
    lightingShaderSSAO = GetShaderLocation(lightingShader, b"ssao")
    lightingShaderCamPos = GetShaderLocation(lightingShader, b"camPos")
    lightingShaderCamInvViewProj = GetShaderLocation(lightingShader, b"camInvViewProj")
    lightingShaderLightDir = GetShaderLocation(lightingShader, b"lightDir")
    lightingShaderSunColor = GetShaderLocation(lightingShader, b"sunColor")
    lightingShaderSunStrength = GetShaderLocation(lightingShader, b"sunStrength")
    lightingShaderSkyColor = GetShaderLocation(lightingShader, b"skyColor")
    lightingShaderSkyStrength = GetShaderLocation(lightingShader, b"skyStrength")
    lightingShaderGroundStrength = GetShaderLocation(lightingShader, b"groundStrength")
    lightingShaderAmbientStrength = GetShaderLocation(lightingShader, b"ambientStrength")
    lightingShaderExposure = GetShaderLocation(lightingShader, b"exposure")
    lightingShaderCamClipNear = GetShaderLocation(lightingShader, b"camClipNear")
    lightingShaderCamClipFar = GetShaderLocation(lightingShader, b"camClipFar")
    
    ssaoShader = LoadShaderVersioned("./resources/post.vs", "./resources/ssao.fs")
    ssaoShaderGBufferNormal = GetShaderLocation(ssaoShader, b"gbufferNormal")
    ssaoShaderGBufferDepth = GetShaderLocation(ssaoShader, b"gbufferDepth")
    ssaoShaderCamView = GetShaderLocation(ssaoShader, b"camView")
    ssaoShaderCamProj = GetShaderLocation(ssaoShader, b"camProj")
    ssaoShaderCamInvProj = GetShaderLocation(ssaoShader, b"camInvProj")
    ssaoShaderCamInvViewProj = GetShaderLocation(ssaoShader, b"camInvViewProj")
    ssaoShaderLightViewProj = GetShaderLocation(ssaoShader, b"lightViewProj")
    ssaoShaderShadowMap = GetShaderLocation(ssaoShader, b"shadowMap")
    ssaoShaderShadowInvResolution = GetShaderLocation(ssaoShader, b"shadowInvResolution")
    ssaoShaderCamClipNear = GetShaderLocation(ssaoShader, b"camClipNear")
    ssaoShaderCamClipFar = GetShaderLocation(ssaoShader, b"camClipFar")
    ssaoShaderLightClipNear = GetShaderLocation(ssaoShader, b"lightClipNear")
    ssaoShaderLightClipFar = GetShaderLocation(ssaoShader, b"lightClipFar")
    ssaoShaderLightDir = GetShaderLocation(ssaoShader, b"lightDir")
    
    blurShader = LoadShaderVersioned("./resources/post.vs", "./resources/blur.fs")
    blurShaderGBufferNormal = GetShaderLocation(blurShader, b"gbufferNormal")
    blurShaderGBufferDepth = GetShaderLocation(blurShader, b"gbufferDepth")
    blurShaderInputTexture = GetShaderLocation(blurShader, b"inputTexture")
    blurShaderCamInvProj = GetShaderLocation(blurShader, b"camInvProj")
    blurShaderCamClipNear = GetShaderLocation(blurShader, b"camClipNear")
    blurShaderCamClipFar = GetShaderLocation(blurShader, b"camClipFar")
    blurShaderInvTextureResolution = GetShaderLocation(blurShader, b"invTextureResolution")
    blurShaderBlurDirection = GetShaderLocation(blurShader, b"blurDirection")

    fxaaShader = LoadShaderVersioned("./resources/post.vs", "./resources/fxaa.fs")
    fxaaShaderInputTexture = GetShaderLocation(fxaaShader, b"inputTexture")
    fxaaShaderInvTextureResolution = GetShaderLocation(fxaaShader, b"invTextureResolution")

    # Onion-skin ghost shader: reuses the skinning vertex shader (so it follows
    # the pose) with a simple translucent flat-shaded fragment shader.
    ghostShader = LoadShaderVersioned("./resources/skinnedBasic.vs", "./resources/ghost.fs")
    ghostShaderColor = GetShaderLocation(ghostShader, b"ghostColor")

    # Objects
    
    groundMesh = GenMeshPlane(20.0, 20.0, 10, 10)
    groundModel = LoadModelFromMesh(groundMesh)
    groundPosition = Vector3(0.0, -0.01, 0.0)
    
    genoModel = LoadGenoModel(b"./resources/Geno.bin")
    genoPosition = Vector3(0.0, 0.0, 0.0)
    
    bindPos, bindRot = GetModelBindPoseAsNumpyArrays(genoModel)
    
    # Animation
    #
    # The current clip can be changed at runtime: drag-and-drop a .bvh onto the
    # window, click "Open File", or press F5 to hot-reload the current file.
    # "Open Folder" loads a folder of .bvh files and reveals the Clip dropdown
    # to pick between them.

    currentBvhPath = "./resources/dance1_subject1.bvh"
    anim = LoadAnimation(currentBvhPath)
    parents = anim['parents']
    localPositions = anim['localPositions']
    globalRotations = anim['globalRotations']
    globalPositions = anim['globalPositions']
    footJoints = anim['footJoints']
    footContacts = anim['footContacts']
    frameTime = anim['frameTime']

    # Clip dropdown state. It lists the .bvh files of a folder opened via the
    # "Open Folder" button and is hidden until then. bvhPaths holds full paths;
    # the dropdown displays their basenames.
    BVH_FOLDER = None
    bvhPaths = []
    clipActivePtr = ffi.new('int*'); clipActivePtr[0] = 0
    clipLastActive = 0
    clipDropdownEdit = False

    # Pending "Open File / Open Folder" requests (set by the UI buttons, applied
    # at the top of the next frame).
    pendingOpenFile = None
    pendingOpenFolder = None

    animationFrame = 0
    
    # Camera
    
    camera = Camera()
    
    rlSetClipPlanes(0.01, 50.0)
    
    # Shadows
    
    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    
    shadowLight = ShadowLight()
    shadowLight.target = Vector3Zero()
    shadowLight.position = Vector3Scale(lightDir, -5.0)
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.width = 5.0
    shadowLight.height = 5.0
    shadowLight.near = 0.01
    shadowLight.far = 10.0
    
    shadowWidth = 1024
    shadowHeight = 1024
    shadowInvResolution = Vector2(1.0 / shadowWidth, 1.0 / shadowHeight)
    shadowMap = LoadShadowMap(shadowWidth, shadowHeight)    
    
    # GBuffer and Render Textures
    
    gbuffer = LoadGBuffer(screenWidth, screenHeight)
    lighted = LoadRenderTexture(screenWidth, screenHeight)
    ssaoFront = LoadRenderTexture(screenWidth, screenHeight)
    ssaoBack = LoadRenderTexture(screenWidth, screenHeight)
    
    # UI

    drawBoneTransformsPtr = ffi.new('bool*'); drawBoneTransformsPtr[0] = False
    skeletonPtr = ffi.new('bool*'); skeletonPtr[0] = False
    footSlidePtr = ffi.new('bool*'); footSlidePtr[0] = True
    onionSkinPtr = ffi.new('bool*'); onionSkinPtr[0] = True
    footSlideResults = []

    # Playback controls

    numFrames = len(localPositions)
    maxFrame = max(numFrames - 1, 0)

    frameFloatPtr = ffi.new('float*'); frameFloatPtr[0] = 0.0   # current frame (fractional, scrubber value)
    speedPtr = ffi.new('float*'); speedPtr[0] = 1.0            # speed multiplier
    modePtr = ffi.new('int*'); modePtr[0] = 0                  # 0 = Loop, 1 = PingPong, 2 = Once

    playing = True
    direction = 1                                             # +1/-1, used by PingPong

    # Go

    while not WindowShouldClose():

        # Animation

        # --- Load a different clip: Open buttons / drag-drop / dropdown / F5 ---
        newPath = None
        singleFile = False                                     # True => exit folder mode (hide dropdown)

        if pendingOpenFolder is not None:                      # "Open Folder" button
            BVH_FOLDER = pendingOpenFolder
            pendingOpenFolder = None
            bvhPaths = ScanBvhFolder(BVH_FOLDER)
            if bvhPaths:
                # if the current clip is in this folder just select it (keep
                # playing); otherwise load the folder's first clip
                idx = next((i for i, p in enumerate(bvhPaths)
                            if os.path.basename(p) == os.path.basename(currentBvhPath)), -1)
                if idx >= 0:
                    clipActivePtr[0] = idx
                    clipLastActive = idx
                else:
                    newPath = bvhPaths[0]

        if pendingOpenFile is not None:                        # "Open File" button
            if pendingOpenFile.lower().endswith(".bvh"):
                newPath = pendingOpenFile
                singleFile = True
            else:
                print("Not a .bvh file:", pendingOpenFile)
            pendingOpenFile = None

        if IsFileDropped():                                    # drag-and-drop
            dropped = LoadDroppedFiles()
            for i in range(dropped.count):
                pth = ffi.string(dropped.paths[i]).decode()
                if pth.lower().endswith(".bvh"):
                    newPath = pth
                    singleFile = True
                    break
            UnloadDroppedFiles(dropped)

        if bvhPaths and clipActivePtr[0] != clipLastActive:    # dropdown selection changed
            clipLastActive = clipActivePtr[0]
            if 0 <= clipActivePtr[0] < len(bvhPaths):
                newPath = bvhPaths[clipActivePtr[0]]

        if IsKeyPressed(KEY_F5):                               # hot-reload current file
            newPath = currentBvhPath
            if BVH_FOLDER is not None:                         # refresh the open folder's list
                bvhPaths = ScanBvhFolder(BVH_FOLDER)

        if newPath is not None:
            try:
                anim = LoadAnimation(newPath)
                parents = anim['parents']
                localPositions = anim['localPositions']
                globalRotations = anim['globalRotations']
                globalPositions = anim['globalPositions']
                footJoints = anim['footJoints']
                footContacts = anim['footContacts']
                frameTime = anim['frameTime']
                currentBvhPath = anim['path']
                frameFloatPtr[0] = 0.0
                direction = 1
                if singleFile:
                    # opening a standalone file exits folder mode -> hide dropdown
                    BVH_FOLDER = None
                    bvhPaths = []
                    clipActivePtr[0] = 0
                    clipLastActive = 0
                    clipDropdownEdit = False
                else:
                    # if the loaded clip belongs to the open folder, point the
                    # dropdown at it (otherwise leave the folder list as-is)
                    idx = next((i for i, p in enumerate(bvhPaths)
                                if os.path.basename(p) == anim['name']), -1)
                    if idx >= 0:
                        clipActivePtr[0] = idx
                        clipLastActive = idx
                SetWindowTitle(b"GenoViewPython - " + anim['name'].encode())
            except Exception as e:
                print("Failed to load '%s': %s" % (newPath, e))

        numFrames = len(localPositions)
        maxFrame = max(numFrames - 1, 0)

        # --- Playback input (keyboard shortcuts) ---
        if IsKeyPressed(KEY_SPACE):
            playing = not playing
        if IsKeyPressed(KEY_RIGHT):                            # step +1 frame
            playing = False
            frameFloatPtr[0] = min(float(maxFrame), float(int(frameFloatPtr[0]) + 1))
        if IsKeyPressed(KEY_LEFT):                             # step -1 frame
            playing = False
            frameFloatPtr[0] = max(0.0, float(int(frameFloatPtr[0]) - 1))
        if IsKeyPressed(KEY_HOME):                             # jump to start
            frameFloatPtr[0] = 0.0
        if IsKeyPressed(KEY_END):                              # jump to end
            frameFloatPtr[0] = float(maxFrame)
        if IsKeyPressed(KEY_UP):                               # faster
            speedPtr[0] = min(4.0, speedPtr[0] + 0.25)
        if IsKeyPressed(KEY_DOWN):                             # slower
            speedPtr[0] = max(0.1, speedPtr[0] - 0.25)

        # --- Advance the playhead ---
        # Advance by real elapsed time / clip frame time so playback runs at the
        # clip's true rate regardless of render FPS or clip FPS (clamped so a
        # hitch can't skip a huge chunk). speed is the user multiplier.
        step = speedPtr[0] * min(GetFrameTime(), 0.1) / frameTime
        if playing and maxFrame > 0:
            if modePtr[0] == 1:                               # PingPong
                frameFloatPtr[0] += step * direction
                if frameFloatPtr[0] >= maxFrame:
                    frameFloatPtr[0] = float(maxFrame); direction = -1
                elif frameFloatPtr[0] <= 0.0:
                    frameFloatPtr[0] = 0.0; direction = 1
            else:                                             # Loop / Once
                direction = 1
                frameFloatPtr[0] += step
                if frameFloatPtr[0] >= maxFrame:
                    if modePtr[0] == 0:                       # Loop: wrap around
                        frameFloatPtr[0] -= maxFrame
                    else:                                     # Once: stop at the end
                        frameFloatPtr[0] = float(maxFrame); playing = False

        # keep the playhead in range (e.g. after scrubbing / mode changes)
        frameFloatPtr[0] = min(float(maxFrame), max(0.0, frameFloatPtr[0]))

        animationFrame = min(maxFrame, int(frameFloatPtr[0]))
        UpdateModelPoseFromNumpyArrays(
            genoModel, bindPos, bindRot,
            globalPositions[animationFrame], globalRotations[animationFrame])

        # Shadow Light Tracks Character
        
        hipPosition = Vector3(*globalPositions[animationFrame][0])
        
        shadowLight.target = Vector3(hipPosition.x, 0.0, hipPosition.z)
        shadowLight.position = Vector3Add(shadowLight.target, Vector3Scale(lightDir, -5.0))

        # Update Camera
        
        camera.update(
            Vector3(hipPosition.x, 0.75, hipPosition.z),
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove(),
            GetFrameTime())
        
        # Render
        
        rlDisableColorBlend()
        
        BeginDrawing()
        
        # Render Shadow Maps
        
        BeginShadowMap(shadowMap, shadowLight)  
        
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNear = rlGetCullDistanceNear()
        lightClipFar = rlGetCullDistanceFar()

        lightClipNearPtr = ffi.new("float*"); lightClipNearPtr[0] = lightClipNear
        lightClipFarPtr = ffi.new("float*"); lightClipFarPtr[0] = lightClipFar
        
        SetShaderValue(shadowShader, shadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadowShader, shadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        groundModel.materials[0].shader = shadowShader
        DrawModel(groundModel, groundPosition, 1.0, WHITE)
        
        genoModel.materials[0].shader = skinnedShadowShader
        DrawModel(genoModel, genoPosition, 1.0, WHITE)
        
        EndShadowMap()
        
        # Render GBuffer
        
        BeginGBuffer(gbuffer, camera.cam3d)
        
        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNear = rlGetCullDistanceNear()
        camClipFar = rlGetCullDistanceFar()

        camClipNearPtr = ffi.new("float*"); camClipNearPtr[0] = camClipNear
        camClipFarPtr = ffi.new("float*"); camClipFarPtr[0] = camClipFar

        specularityPtr = ffi.new('float*'); specularityPtr[0] = 0.5
        glossinessPtr = ffi.new('float*'); glossinessPtr[0] = 10.0
        
        SetShaderValue(basicShader, basicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)        
        
        groundModel.materials[0].shader = basicShader
        DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))
        
        genoModel.materials[0].shader = skinnedBasicShader
        DrawModel(genoModel, genoPosition, 1.0, ORANGE)       
        
        EndGBuffer(screenWidth, screenHeight)
        
        # Render SSAO and Shadows
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(ssaoShader)
        
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferDepth, gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamView, camView)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamProj, camProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvViewProj, camInvViewProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderLightViewProj, lightViewProj)
        SetShaderValueShadowMap(ssaoShader, ssaoShaderShadowMap, shadowMap)
        SetShaderValue(ssaoShader, ssaoShaderShadowInvResolution, ffi.addressof(shadowInvResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(ssaoShader, ssaoShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        
        ClearBackground(WHITE)
        
        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0.0, 0.0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
        
        # Blur Horizontal
        
        BeginTextureMode(ssaoBack)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(1.0 / ssaoFront.texture.width, 1.0 / ssaoFront.texture.height)
        
        SetShaderValueTexture(blurShader, blurShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(blurShader, blurShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoFront.texture)
        SetShaderValueMatrix(blurShader, blurShaderCamInvProj, camInvProj)
        SetShaderValue(blurShader, blurShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderInvTextureResolution, ffi.addressof(blurInvTextureResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoBack.texture,
            Rectangle(0, 0, ssaoBack.texture.width, -ssaoBack.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Blur Vertical
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(0.0, 1.0)
        
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoBack.texture)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Light GBuffer
        
        BeginTextureMode(lighted)
        
        BeginShaderMode(lightingShader)
        
        sunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        sunStrengthPtr = ffi.new('float*'); sunStrengthPtr[0] = 0.25
        skyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        skyStrengthPtr = ffi.new('float*'); skyStrengthPtr[0] = 0.15
        groundStrengthPtr = ffi.new('float*'); groundStrengthPtr[0] = 0.1
        ambientStrengthPtr = ffi.new('float*'); ambientStrengthPtr[0] = 1.0
        exposurePtr = ffi.new('float*'); exposurePtr[0] = 0.9
        
        SetShaderValueTexture(lightingShader, lightingShaderGBufferColor, gbuffer.color)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(lightingShader, lightingShaderSSAO, ssaoFront.texture)
        SetShaderValue(lightingShader, lightingShaderCamPos, ffi.addressof(camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, lightingShaderCamInvViewProj, camInvViewProj)
        SetShaderValue(lightingShader, lightingShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunColor, ffi.addressof(sunColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunStrength, sunStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderSkyColor, ffi.addressof(skyColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSkyStrength, skyStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderGroundStrength, groundStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderAmbientStrength, ambientStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderExposure, exposurePtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        ClearBackground(RAYWHITE)
        
        DrawTextureRec(
            gbuffer.color,
            Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()        
        
        # Debug Draw
        
        BeginMode3D(camera.cam3d)

        if onionSkinPtr[0]:
            DrawOnionGhosts(
                genoModel, bindPos, bindRot, globalPositions, globalRotations,
                animationFrame, len(localPositions) - 1, ghostShader, ghostShaderColor)

        if skeletonPtr[0]:
            DrawSkeletonOverlay(globalPositions[animationFrame], parents)

        if drawBoneTransformsPtr[0]:
            DrawSkeleton(
                globalPositions[animationFrame],
                globalRotations[animationFrame],
                parents, GRAY)

        footSlideResults = []
        if footSlidePtr[0]:
            for (label, j) in footJoints:
                pos, hSpeed, grounded, sliding, penetrating = MeasureFootSlide(
                    globalPositions, animationFrame, j, footContacts[j])
                DrawFootSlide(pos, hSpeed, grounded, sliding, penetrating)
                footSlideResults.append((label, hSpeed, grounded, sliding, penetrating))

        EndMode3D()

        EndTextureMode()
        
        # Render Final with FXAA
        
        BeginShaderMode(fxaaShader)

        fxaaInvTextureResolution = Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
        
        SetShaderValueTexture(fxaaShader, fxaaShaderInputTexture, lighted.texture)
        SetShaderValue(fxaaShader, fxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)
        
        DrawTextureRec(
            lighted.texture,
            Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()
  
        # UI

        rlEnableColorBlend()

        # While the clip dropdown is open, lock the other controls so its list
        # doesn't leak clicks through to whatever sits beneath it.
        if clipDropdownEdit:
            GuiLock()

        GuiGroupBox(Rectangle(20, 10, 190, 180), b"Camera")

        GuiLabel(Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
        GuiLabel(Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
        GuiLabel(Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
        GuiLabel(Rectangle(30, 80, 150, 20), b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z))
        GuiLabel(Rectangle(30, 100, 150, 20), b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z))
        GuiLabel(Rectangle(30, 120, 150, 20), b"Azimuth: %5.3f" % camera.azimuth)
        GuiLabel(Rectangle(30, 140, 150, 20), b"Altitude: %5.3f" % camera.altitude)
        GuiLabel(Rectangle(30, 160, 150, 20), b"Distance: %5.3f" % camera.distance)
  
        GuiGroupBox(Rectangle(screenWidth - 260, 10, 240, 118), b"Rendering")

        GuiCheckBox(Rectangle(screenWidth - 250, 20, 20, 20), b"Skeleton", skeletonPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 46, 20, 20), b"Draw Transforms", drawBoneTransformsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 72, 20, 20), b"Foot Slide", footSlidePtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 98, 20, 20), b"Onion Skin", onionSkinPtr)

        # Foot-slide readout (live per-foot horizontal speed; red while sliding)
        if footSlidePtr[0] and footSlideResults:
            boxH = 30 + 18 * len(footSlideResults)
            GuiGroupBox(Rectangle(screenWidth - 260, 138, 240, boxH), b"Foot Slide (m/s)")
            for k, (label, hSpeed, grounded, sliding, penetrating) in enumerate(footSlideResults):
                if penetrating:
                    state = b"BELOW FLOOR"
                elif sliding:
                    state = b"SLIDING"
                elif grounded:
                    state = b"planted"
                else:
                    state = b"air"
                GuiLabel(Rectangle(screenWidth - 250, 148 + 18 * k, 220, 18),
                         b"%-7s %5.3f  %s" % (label.encode(), hSpeed, state))

        # Playback controls

        panelX = 20
        panelW = screenWidth - 40
        panelH = 86
        panelY = screenHeight - panelH - 10
        GuiGroupBox(Rectangle(panelX, panelY, panelW, panelH), b"Playback")

        # Timeline scrubber + frame counter
        sx = panelX + 16
        scrubW = panelW - 32 - 140
        scrubY = panelY + 16
        GuiSliderBar(Rectangle(sx, scrubY, scrubW, 16), b"", b"", frameFloatPtr, 0.0, float(maxFrame))
        GuiLabel(Rectangle(sx + scrubW + 10, scrubY - 2, 140, 20),
                 b"Frame %d / %d" % (animationFrame, maxFrame))

        # Transport buttons
        by = panelY + 46
        bh = 26
        bx = sx
        if GuiButton(Rectangle(bx, by, 36, bh), b"|<"):                       # jump to start
            frameFloatPtr[0] = 0.0
        bx += 40
        if GuiButton(Rectangle(bx, by, 36, bh), b"<"):                        # step back
            playing = False
            frameFloatPtr[0] = max(0.0, float(int(frameFloatPtr[0]) - 1))
        bx += 40
        if GuiButton(Rectangle(bx, by, 74, bh), b"Pause" if playing else b"Play"):
            playing = not playing
        bx += 78
        if GuiButton(Rectangle(bx, by, 36, bh), b">"):                        # step forward
            playing = False
            frameFloatPtr[0] = min(float(maxFrame), float(int(frameFloatPtr[0]) + 1))
        bx += 40
        if GuiButton(Rectangle(bx, by, 36, bh), b">|"):                       # jump to end
            frameFloatPtr[0] = float(maxFrame)

        # Speed multiplier
        GuiSliderBar(Rectangle(bx + 110, by + 4, 150, 16), b"Speed",
                     b"%.2fx" % speedPtr[0], speedPtr, 0.1, 4.0)

        # Loop / PingPong / Once
        GuiComboBox(Rectangle(panelX + panelW - 130, by, 114, bh),
                    b"Loop;PingPong;Once", modePtr)

        # Clip controls + Open buttons (drawn last so the open list overlays all)
        GuiUnlock()
        cx = screenWidth / 2
        GuiLabel(Rectangle(cx - 200, 14, 400, 18),
                 b"Drag a .bvh onto the window   -   F5 to reload")

        # Open File / Open Folder, placed just below the Camera box (x: 20..210)
        if GuiButton(Rectangle(20, 198, 92, 26), b"Open File"):
            pendingOpenFile = NativeChoose(folder=False)
        if GuiButton(Rectangle(118, 198, 92, 26), b"Open Folder"):
            pendingOpenFolder = NativeChoose(folder=True)

        # The Clip dropdown only appears once a folder of .bvh files is opened.
        if bvhPaths:
            GuiLabel(Rectangle(cx - 175, 40, 40, 18), b"Clip")
            clipText = b";".join(os.path.basename(p).encode() for p in bvhPaths)
            if GuiDropdownBox(Rectangle(cx - 130, 36, 260, 26),
                              clipText, clipActivePtr, clipDropdownEdit):
                clipDropdownEdit = not clipDropdownEdit

        EndDrawing()

    UnloadRenderTexture(lighted)
    UnloadRenderTexture(ssaoBack)
    UnloadRenderTexture(ssaoFront)
    UnloadRenderTexture(lighted)
    UnloadGBuffer(gbuffer)

    UnloadShadowMap(shadowMap)
    
    UnloadModel(genoModel)
    UnloadModel(groundModel)
    
    UnloadShader(ghostShader)
    UnloadShader(fxaaShader)
    UnloadShader(blurShader)    
    UnloadShader(ssaoShader) 
    UnloadShader(lightingShader)    
    UnloadShader(basicShader)
    UnloadShader(skinnedBasicShader)
    UnloadShader(skinnedShadowShader)
    UnloadShader(shadowShader)
    
    CloseWindow()