use std::fmt;
use std::io;

use bevy::app::AppExit;
use bevy::asset::AssetPlugin;
use bevy::core_pipeline::core_3d::Camera3d;
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::diagnostic::DiagnosticsPlugin;
use bevy::input::InputPlugin;
use bevy::log::LogPlugin;
use bevy::pbr::{PbrPlugin, MaterialMeshBundle};
use bevy::prelude::*;
use bevy::render::{RenderPlugin, settings::{WgpuSettings, WgpuFeatures, Backends}};
use bevy::render::mesh::shape;
use bevy::transform::TransformPlugin;

use bevy_capture::prelude::*;

use image::{
    codecs::gif::{GifEncoder, Repeat},
    Delay, Frame, ImageError, RgbaImage,
};
use image::imageops;
use crate::display::{display_gif, DisplayError};
use crate::embed::Point3D;

/// Custom error type for video-related operations.
#[derive(Debug)]
pub enum VideoError {
    Io(std::io::Error),
    Image(ImageError),
    Display(DisplayError),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Io(e) => write!(f, "IO error: {}", e),
            VideoError::Image(e) => write!(f, "Image error: {}", e),
            VideoError::Display(e) => write!(f, "Display error: {}", e),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<std::io::Error> for VideoError {
    fn from(e: std::io::Error) -> Self {
        VideoError::Io(e)
    }
}

impl From<ImageError> for VideoError {
    fn from(e: ImageError) -> Self {
        VideoError::Image(e)
    }
}

impl From<DisplayError> for VideoError {
    fn from(e: DisplayError) -> Self {
        VideoError::Display(e)
    }
}

/// Stores all frames captured from the "render" process, plus counters.
#[derive(Resource)]
struct RenderResources {
    frames: Vec<Frame>,
    frame_count: usize,
    total_frames: usize,
}

/// Holds the 3D points we want to display.
#[derive(Resource)]
struct PointCloudData {
    points: Vec<Point3D>,
}

/// Renders a spinning 3D view of the given points, captures frames into a GIF,
/// and displays that GIF in the terminal.
pub fn render(points: Vec<Point3D>) -> Result<(), VideoError> {
    // Number of frames we want to capture for the GIF.
    const TOTAL_FRAMES: usize = 60;

    // Build a Bevy app with the necessary plugins for headless 3D rendering
    let mut app = App::new();

    // Configure Vulkan/Metal for headless GPU access
    let wgpu_settings = WgpuSettings {
        backends: Some(Backends::VULKAN),
        features: WgpuFeatures::empty(),
        ..default()
    };

    app.insert_resource(wgpu_settings)
        // Add minimal set of plugins needed for headless 3D rendering
        .add_plugin(LogPlugin::default())
        .add_plugin(TransformPlugin::default())
        .add_plugin(InputPlugin::default())
        .add_plugin(DiagnosticsPlugin::default())
        .add_plugin(AssetPlugin::default())
        .add_plugin(RenderPlugin::default())
        .add_plugin(CorePipelinePlugin::default())
        .add_plugin(PbrPlugin::default())
        // Add the capture plugin
        .add_plugin(CapturePlugin::default())
        // Resource that collects frames
        .insert_resource(RenderResources {
            frames: Vec::with_capacity(TOTAL_FRAMES),
            frame_count: 0,
            total_frames: TOTAL_FRAMES,
        })
        // Resource with user data
        .insert_resource(PointCloudData { points })
        // Set up scene once at startup
        .add_startup_system(setup)
        // Rotate the camera each frame
        .add_system(rotate_camera)
        // Capture each frame
        .add_system(capture_frame)
        // Check if we've finished collecting all frames
        .add_system(check_finished);

    // Run the Bevy main loop (blocks until exit).
    app.run();

    // After app exits, collect the frames from the world resource into a GIF.
    let render_resources = app.world()
        .get_resource::<RenderResources>()
        .expect("RenderResources missing from World");

    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        // Loop forever
        encoder.set_repeat(Repeat::Infinite)?;
        // Encode all frames
        for frame in &render_resources.frames {
            encoder.encode_frame(frame.clone())?;
        }
    }

    // Finally, display the resulting GIF in the terminal.
    display_gif(&gif_data)?;

    Ok(())
}

/// Sets up the scene: spawns a camera (with capture), light, spheres for points, and XYZ axes.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    point_cloud: Res<PointCloudData>,
) {
    // Camera that we'll use for capturing frames.
    //
    // - `Camera3d::default()` configures a 3D camera pipeline.
    // - `CaptureCamera::default()` from bevy_capture marks this camera for screenshot capture.
    // - We also add a normal `Transform` and `GlobalTransform`.
    commands.spawn((
        Camera3d::default(),
        CaptureCamera::default(), // from bevy_capture
        Transform::from_xyz(0.0, 0.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
    ));

    // A point light to illuminate the scene.
    commands.spawn((
        PointLight {
            intensity: 1800.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
        GlobalTransform::default(),
    ));

    // Spawn small spheres for each point in the data
    for point in &point_cloud.points {
        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 0.05,
                sectors: 16,
                stacks: 16,
            })),
            material: materials.add(StandardMaterial {
                // Use sRGB for correct color
                base_color: Color::srgb_u8(
                    point.color.0[0],
                    point.color.0[1],
                    point.color.0[2]
                ),
                unlit: false, // or true if you want no lighting
                ..default()
            }),
            transform: Transform::from_xyz(point.pos.x, point.pos.y, point.pos.z),
            ..default()
        });
    }

    // Draw X/Y/Z axes as colored cylinders, each 10 units long
    let axes = [
        (Vec3::X, Color::RED),
        (Vec3::Y, Color::GREEN),
        (Vec3::Z, Color::BLUE),
    ];
    for (direction, color) in axes {
        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cylinder {
                radius: 0.02,
                height: 10.0,
                ..default()
            })),
            material: materials.add(StandardMaterial {
                base_color: color,
                unlit: true,
                ..default()
            }),
            transform: Transform::default().looking_to(direction, Vec3::Y),
            ..default()
        });
    }
}

/// System to rotate the camera around the origin across `total_frames`.
fn rotate_camera(
    render_resources: Res<RenderResources>,
    mut camera_query: Query<&mut Transform, (With<Camera3d>, With<CaptureCamera>)>,
) {
    // If done capturing, no need to move camera
    if render_resources.frame_count >= render_resources.total_frames {
        return;
    }

    // Calculate fraction of full rotation based on how many frames we've captured so far
    let fraction = render_resources.frame_count as f32 / render_resources.total_frames as f32;
    let angle = fraction * std::f32::consts::TAU;

    // Move camera in a circle around the origin, slightly above on Y
    if let Ok(mut transform) = camera_query.get_single_mut() {
        transform.translation = Vec3::new(15.0 * angle.cos(), 5.0, 15.0 * angle.sin());
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

/// Captures the current camera image each frame. We store it as a `Frame` for GIF encoding.
fn capture_frame(
    mut render_resources: ResMut<RenderResources>,
    camera_query: Query<&CaptureCamera, With<Camera3d>>,
    images: Res<Assets<Image>>,
) {
    // If we've already captured all desired frames, skip
    if render_resources.frame_count >= render_resources.total_frames {
        return;
    }

    // We only have one capturing camera. Try to get it.
    if let Ok(capture_cam) = camera_query.get_single() {
        // Attempt to retrieve the latest captured frame from GPU
        if let Some(captured) = capture_cam.capture_image(&images) {
            let width = captured.texture_descriptor.size.width;
            let height = captured.texture_descriptor.size.height;
            let raw_data = &captured.data;

            // The GPU returns RGBA bytes.
            // Convert that to a RgbaImage from the `image` crate:
            let mut rgba_image = match RgbaImage::from_raw(width, height, raw_data.clone()) {
                Some(img) => img,
                None => {
                    eprintln!("Failed to create RgbaImage from raw GPU data!");
                    return;
                }
            };

            // Many GPU captures come in "upside-down" (Y=0 at top). Flip vertically.
            imageops::flip_vertical_in_place(&mut rgba_image);

            // Insert it into an animated Frame, ~16ms (60 FPS).
            let frame = Frame::from_parts(rgba_image, 0, 0, Delay::from_numer_denom_ms(16, 1));

            // Store in our resource to assemble into a GIF later
            render_resources.frames.push(frame);
            render_resources.frame_count += 1;
        }
    }
}

/// Checks if we've captured all frames, and if so requests an exit from the App.
fn check_finished(
    render_resources: Res<RenderResources>,
    mut exit: EventWriter<AppExit>,
) {
    if render_resources.frame_count >= render_resources.total_frames {
        exit.send(AppExit {});
    }
}
