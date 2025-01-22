use std::fmt;
use std::io;
use std::fs::File;
use std::path::PathBuf;

use bevy::app::AppExit;
use bevy::asset::AssetPlugin;
use bevy::core_pipeline::core_3d::Camera3d;
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::diagnostic::DiagnosticsPlugin;
use bevy::input::InputPlugin;
use bevy::log::LogPlugin;
use bevy::pbr::{PbrPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::RenderPlugin;
use bevy::transform::TransformPlugin;
use bevy::render::mesh::Indices;
use bevy::render::render_asset::RenderAssetUsages;

use bevy_capture::{Capture, CaptureBundle, CapturePlugin};
use bevy_capture::encoder::frames::FramesEncoder;

use image::{
    codecs::gif::{GifEncoder, Repeat},
    imageops, Delay, Frame, ImageError, RgbaImage,
};

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
    // Number of frames we want to capture for the GIF
    const TOTAL_FRAMES: usize = 60;

    let mut app = App::new();

    // Build up the minimal set of plugins for 3D rendering and capturing
    app.add_plugins((
        // Minimal base
        MinimalPlugins,
        // Logging, transforms, input, assets, rendering, PBR, etc.
        LogPlugin::default(),
        TransformPlugin::default(),
        InputPlugin::default(),
        DiagnosticsPlugin::default(),
        AssetPlugin::default(),
        RenderPlugin::default(),
        CorePipelinePlugin::default(),
        PbrPlugin::default(),
        // Capture plugin for screenshot
        CapturePlugin,
    ))
    // Insert resources
    .insert_resource(RenderResources {
        frames: Vec::with_capacity(TOTAL_FRAMES),
        frame_count: 0,
        total_frames: TOTAL_FRAMES,
    })
    .insert_resource(PointCloudData { points })
    // Startup: create camera, lights, geometry
    .add_systems(Startup, setup)
    // Main loop: rotate camera, capture frames, exit if done
    .add_systems(Update, (rotate_camera, capture_frame, check_finished));

    // Run the Bevy app until we exit (after capturing all frames)
    app.run();

    // After the app exits, retrieve the frames from RenderResources and build a GIF
    let render_resources = app
        .world()
        .get_resource::<RenderResources>()
        .expect("RenderResources missing from the World");

    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        encoder.set_repeat(Repeat::Infinite)?;
        // Append all frames
        for frame in &render_resources.frames {
            encoder.encode_frame(frame.clone())?;
        }
    }

    // Display the resulting GIF in the terminal
    display_gif(&gif_data)?;

    Ok(())
}

/// Sets up the scene: spawns camera, lights, point-cloud spheres, and X/Y/Z axes.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    point_cloud: Res<PointCloudData>,
) {
    // Camera with capturing
    commands.spawn((
        Camera3d {
            ..Default::default()
        },
        Camera::default().target_headless(512, 512, &mut images),
        CaptureBundle::default(),
        Transform::from_xyz(0.0, 0.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
    ));

    // Light
    commands.spawn((
        PointLight {
            intensity: 1800.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
        GlobalTransform::default(),
    ));

    // For each embedded point, spawn a small sphere
    for point in &point_cloud.points {
        commands.spawn((
            // A sphere mesh
            Mesh3d(meshes.add(create_sphere_mesh(0.05, 16, 16))),
            // Painted standard material
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb_u8(
                    point.color.0[0],
                    point.color.0[1],
                    point.color.0[2],
                ),
                unlit: false,
                ..default()
            })),
            // Position from the point data
            Transform::from_xyz(point.pos.x, point.pos.y, point.pos.z),
            GlobalTransform::default(),
        ));
    }

    // Build three colored cylinders for X/Y/Z axes
    let axes = [
        (Vec3::X, Color::srgb(1.0, 0.0, 0.0)), // red
        (Vec3::Y, Color::srgb(0.0, 1.0, 0.0)), // green
        (Vec3::Z, Color::srgb(0.0, 0.0, 1.0)), // blue
    ];

    for (direction, color) in axes {
        commands.spawn((
            Mesh3d(meshes.add(create_cylinder_mesh(0.02, 10.0, 16))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                unlit: true,
                ..default()
            })),
            // Start at origin, orient along direction
            Transform::default().looking_to(direction, Vec3::Y),
            GlobalTransform::default(),
        ));
    }
}

/// Rotates the camera around the origin over `total_frames`.
fn rotate_camera(
    render_resources: Res<RenderResources>,
    mut camera_query: Query<&mut Transform, (With<Camera3d>, With<Capture>)>,
) {
    // Stop once all frames have been captured
    if render_resources.frame_count >= render_resources.total_frames {
        return;
    }

    let fraction = render_resources.frame_count as f32 / render_resources.total_frames as f32;
    let angle = fraction * std::f32::consts::TAU;

    if let Ok(mut transform) = camera_query.get_single_mut() {
        transform.translation = Vec3::new(15.0 * angle.cos(), 5.0, 15.0 * angle.sin());
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

/// Captures the camera image each frame and stores it in a buffer for the GIF.
fn capture_frame(
    mut render_resources: ResMut<RenderResources>,
    mut capture_query: Query<&mut Capture, With<Camera3d>>,
) {
    if render_resources.frame_count >= render_resources.total_frames {
        return;
    }

    // Start capturing if not already capturing
    if let Ok(mut capture) = capture_query.get_single_mut() {
        if !capture.is_capturing() {
            let file = File::create("output.gif").expect("Failed to create output file");
            let encoder = GifEncoder::new(file)
                .with_repeat(Repeat::Infinite);
            capture.start(encoder);
        }
        render_resources.frame_count += 1;
    }
}


/// Exits the Bevy app once we've captured enough frames for the GIF.
fn check_finished(render_resources: Res<RenderResources>, mut exit: EventWriter<AppExit>) {
    if render_resources.frame_count >= render_resources.total_frames {
        exit.send(AppExit::Success);
    }
}

/// Creates a simple sphere mesh (latitude-longitude) with the given radius/segments/rings.
fn create_sphere_mesh(radius: f32, segments: u32, rings: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for lat_i in 0..=rings {
        let lat = lat_i as f32 * std::f32::consts::PI / rings as f32;
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();

        for long_i in 0..=segments {
            let long = long_i as f32 * 2.0 * std::f32::consts::PI / segments as f32;
            let sin_long = long.sin();
            let cos_long = long.cos();

            let x = cos_long * sin_lat;
            let y = cos_lat;
            let z = sin_long * sin_lat;

            positions.push([radius * x, radius * y, radius * z]);
            normals.push([x, y, z]);
            // approximate UVs
            uvs.push([long / (2.0 * std::f32::consts::PI), 1.0 - (lat / std::f32::consts::PI)]);
        }
    }

    // Construct indices
    for lat_i in 0..rings {
        for long_i in 0..segments {
            let first = lat_i * (segments + 1) + long_i;
            let second = first + segments + 1;
            indices.push(first);
            indices.push(second);
            indices.push(first + 1);
            indices.push(second);
            indices.push(second + 1);
            indices.push(first + 1);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Creates a simple cylinder mesh aligned along +Y (height in Y).
/// `segments` is how many subdivisions for the circular cross-section.
fn create_cylinder_mesh(radius: f32, height: f32, segments: u32) -> Mesh {
    let half_h = height / 2.0;

    // We'll build a top circle and bottom circle, plus side quads
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    // We generate the top/bottom circles
    // We'll store ring of vertices for top, ring for bottom, plus a center vertex for each.
    let top_center_idx = 0;
    let bottom_center_idx = 1;
    let start_ring_top = 2;
    let start_ring_bottom = 2 + segments;

    // We'll place a dummy at index 0 for top center, index 1 for bottom center
    positions.push([0.0, half_h, 0.0]);  // top center
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    positions.push([0.0, -half_h, 0.0]); // bottom center
    normals.push([0.0, -1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    // build ring for top
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let (sin_a, cos_a) = (angle.sin(), angle.cos());
        let x = radius * cos_a;
        let z = radius * sin_a;

        positions.push([x, half_h, z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([(cos_a * 0.5) + 0.5, (sin_a * 0.5) + 0.5]);
    }

    // build ring for bottom
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let (sin_a, cos_a) = (angle.sin(), angle.cos());
        let x = radius * cos_a;
        let z = radius * sin_a;

        positions.push([x, -half_h, z]);
        normals.push([0.0, -1.0, 0.0]);
        uvs.push([(cos_a * 0.5) + 0.5, (sin_a * 0.5) + 0.5]);
    }

    // Indices for top circle
    for i in 0..segments {
        let ring_idx = start_ring_top + i;
        let next = start_ring_top + ((i + 1) % segments);
        indices.push(top_center_idx);
        indices.push(ring_idx);
        indices.push(next);
    }

    // Indices for bottom circle
    for i in 0..segments {
        let ring_idx = start_ring_bottom + i;
        let next = start_ring_bottom + ((i + 1) % segments);
        indices.push(bottom_center_idx);
        indices.push(next);
        indices.push(ring_idx);
    }

    // Now the side. We'll store these in separate arrays because
    // the top ring & bottom ring have different normals. We want side normals outward.
    let side_start = positions.len() as u32;

    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let (sin_a, cos_a) = (angle.sin(), angle.cos());
        let x = radius * cos_a;
        let z = radius * sin_a;

        // top vertex
        positions.push([x, half_h, z]);
        normals.push([cos_a, 0.0, sin_a]);
        uvs.push([i as f32 / segments as f32, 0.0]);

        // bottom vertex
        positions.push([x, -half_h, z]);
        normals.push([cos_a, 0.0, sin_a]);
        uvs.push([i as f32 / segments as f32, 1.0]);
    }

    // side indices
    for i in 0..segments {
        let top0 = side_start + i * 2;
        let bot0 = top0 + 1;
        let top1 = side_start + ((i + 1) % segments) * 2;
        let bot1 = top1 + 1;

        // quad for side: (top0, bot0, top1), (top1, bot0, bot1)
        indices.push(top0);
        indices.push(bot0);
        indices.push(top1);
        indices.push(top1);
        indices.push(bot0);
        indices.push(bot1);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
