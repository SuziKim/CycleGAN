
--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
if opt.DATA_ROOT then
  opt.data = paths.concat(opt.DATA_ROOT, opt.phase)
else
  print(os.getenv('DATA_ROOT'))
  opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)
end

if not paths.dirp(opt.data) then
    error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.nc -- input channels

local function loadSingleTMap(path)
    local im = image.load(path, 2, 'float')
    return im
end

local function loadSingleCText(path)
    local txtFile = io.open(path)
    local txt = torch.Tensor(input_nc, opt.cTextSize):fill(0)

    local txtFileCursor = 1
    if txtFile then
      for line in txtFile:lines() do
        txt[1][txtFileCursor] = tonumber(line)
        txtFileCursor = txtFileCursor + 1
      end
    end
    return txt
end

--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded
-- [CityCraft] function to load terrain map
local trainHook_tMapData = function(self, path)
   collectgarbage()
   local data = loadSingleTMap(path)
   return data
end

-- [CityCraft] function to load city property text
local trainHook_cTextData = function(self, path)
    collectgarbage()
    local data = loadSingleCText(path)
    return data
end

trainLoader = dataLoader{
    paths = {opt.data},
    split = 100,
    serial_batches = opt.serial_batches,
    verbose = true
 }

-- [CityCraft] Separate Loading Data
if opt.data:sub(-4) == "TMap" then
  trainHook = trainHook_tMapData
  trainLoadSize = loadSize{input_nc, opt.tMapSize, opt.tMapSize}
else
  trainHook = trainHook_cTextData
  trainLoadSize = loadSize{input_nc, opt.cTextSize}
end

trainLoader.sampleHookTrain = trainHook
trainLoader.loadSize = trainLoadSize
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
