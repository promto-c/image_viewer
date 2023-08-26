import PyOpenColorIO  as OCIO

# config_path = 'config.ocio'

# config = OCIO.Config.CreateFromFile(config_path)
config = OCIO.GetCurrentConfig()

display = config.getDefaultDisplay()
view = config.getDefaultView(display)
# look = config.getDisplayViewLooks(display, view)[0] if config.getDisplayViewLooks(display, view) else None
# print(look)
transform = OCIO.DisplayViewTransform()
transform.setSrc(OCIO.ROLE_SCENE_LINEAR)
transform.setDisplay(display)
transform.setView(view)

vpt = OCIO.LegacyViewingPipeline()
vpt.setDisplayViewTransform(transform)
# if look:
#     vpt.setLooksOverrideEnabled(True)
#     vpt.setLooksOverride(look)

processor = vpt.getProcessor(config, config.getCurrentContext())
gpu = processor.getDefaultGPUProcessor()

shaderDesc = OCIO.GpuShaderDesc.CreateShaderDesc()
shaderDesc.setLanguage(OCIO.GPU_LANGUAGE_GLSL_1_3)
shaderDesc.setFunctionName("OCIODisplay")
# shaderDesc.setResourcePrefix("ocio_")

# shaderDesc.setTextureMaxWidth(1)
gpu.extractGpuShaderInfo(shaderDesc)
print(shaderDesc.getShaderText())
