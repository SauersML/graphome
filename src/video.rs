use std::fmt;
use std::io;
use bevy::prelude::*;
use image::{
    ImageBuffer,
    Rgb,
    RgbaImage,
    DynamicImage,
    ImageError,
    Frame,
    Delay,
    codecs::gif::{GifEncoder, Repeat},
};
use crate::embed::Point3D;
use crate::display::{display_gif, DisplayError};

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

#[derive(Resource)]
struct RenderResources {
    frames: Vec<Frame>,
    frame_count: usize,
    total_frames: usize,
}

#[derive(Resource)]
struct PointCloudData {
    points: Vec<Point3D>,
}

pub fn render(points: Vec<Point3D>) -> Result<(), VideoError> {
    const TOTAL_FRAMES: usize = 60;

    let mut app = App::new();
    
    app.add_plugins((
        MinimalPlugins,
        RenderPlugin::default(),
        AssetPlugin::default()
    ))
    .insert_resource(RenderResources {
        frames: Vec::with_capacity(TOTAL_FRAMES),
        frame_count: 0,
        total_frames: TOTAL_FRAMES,
    })
    .insert_resource(PointCloudData { points })
    .add_systems(Startup, setup)
    .add_systems(Update, (rotate_scene, capture_frame));

    app.run();

    let render_resources = app.world().resource::<RenderResources>();
    
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        encoder.set_repeat(Repeat::Infinite)?;
        for frame in &render_resources.frames {
            encoder.encode_frame(frame.clone())?;
        }
    }

    display_gif(&gif_data)?;

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    point_cloud: Res<PointCloudData>,
) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 15.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    for point in &point_cloud.points {
        commands.spawn(PbrBundle {
            mesh: Mesh3d(meshes.add(Mesh::from(bevy::prelude::shape::UVSphere {
                radius: 0.05,
                ..default()
            }))),
            material: MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::rgb_u8(
                    point.color.0[0],
                    point.color.0[1],
                    point.color.0[2]
                ),
                unlit: true,
                ..default()
            })),
            transform: Transform::from_xyz(
                point.pos.x,
                point.pos.y,
                point.pos.z,
            ),
            ..default()
        });
    }

    let axes_data = [
        (Vec3::X, Color::rgb(1.0, 0.0, 0.0)),
        (Vec3::Y, Color::rgb(0.0, 1.0, 0.0)),
        (Vec3::Z, Color::rgb(0.0, 0.0, 1.0)),
    ];

    for (direction, color) in axes_data {
        commands.spawn(PbrBundle {
            mesh: Mesh3d(meshes.add(Mesh::from(bevy::prelude::shape::Cylinder {
                radius: 0.02,
                height: 10.0,
                ..default()
            }))),
            material: MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                unlit: true,
                ..default()
            })),
            transform: Transform::from_xyz(0.0, 0.0, 0.0)
                .looking_to(direction, Vec3::Y),
            ..default()
        });
    }
}

fn rotate_scene(
    time: Res<Time>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
    mut render_resources: ResMut<RenderResources>,
) {
    if render_resources.frame_count >= render_resources.total_frames {
        return;
    }

    let angle = (render_resources.frame_count as f32 / render_resources.total_frames as f32) 
        * 2.0 * std::f32::consts::PI;

    for mut transform in camera_query.iter_mut() {
        transform.translation = Vec3::new(
            15.0 * angle.cos(),
            5.0,
            15.0 * angle.sin(),
        );
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

fn capture_frame(
    mut render_resources: ResMut<RenderResources>
) {
    if render_resources.frame_count >= render_resources.total_frames {
        return;
    }

    let image_buffer = ImageBuffer::new(1600, 1200);
    let rgba_image = DynamicImage::ImageRgb8(image_buffer).to_rgba8();
    
    let frame = Frame::from_parts(
        rgba_image,
        0,
        0,
        Delay::from_numer_denom_ms(16, 1)
    );

    render_resources.frames.push(frame);
    render_resources.frame_count += 1;
}
